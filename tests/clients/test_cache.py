import os
from dataclasses import dataclass
from unittest.mock import patch

import pydantic
import pytest
from cachetools import LRUCache
from diskcache import FanoutCache

from dspy.clients.cache import _MAGIC_HEADER, Cache


@dataclass
class DummyResponse:
    message: str
    usage: dict


@pytest.fixture
def cache_config(tmp_path):
    """Default cache configuration."""
    return {
        "enable_disk_cache": True,
        "enable_memory_cache": True,
        "disk_cache_dir": str(tmp_path),
        "disk_size_limit_bytes": 1024 * 1024,  # 1MB
        "memory_max_entries": 100,
    }


@pytest.fixture
def cache(cache_config):
    """Create a cache instance with the default configuration."""
    return Cache(**cache_config)


def test_initialization(tmp_path):
    """Test different cache initialization configurations."""
    # Test memory-only cache
    memory_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir="",
        disk_size_limit_bytes=0,
        memory_max_entries=50,
    )
    assert isinstance(memory_cache.memory_cache, LRUCache)
    assert memory_cache.memory_cache.maxsize == 50
    assert memory_cache.disk_cache == {}

    # Test disk-only cache
    disk_cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024,
        memory_max_entries=0,
    )
    assert isinstance(disk_cache.disk_cache, FanoutCache)
    assert disk_cache.memory_cache == {}

    # Test disabled cache
    disabled_cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        disk_cache_dir="",
        disk_size_limit_bytes=0,
        memory_max_entries=0,
    )
    assert disabled_cache.memory_cache == {}
    assert disabled_cache.disk_cache == {}


def test_cache_key_generation(cache):
    """Test cache key generation with different types of inputs."""
    # Test with simple dictionary
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}
    key = cache.cache_key(request)
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hash is 64 characters

    # Test with pydantic model
    class TestModel(pydantic.BaseModel):
        name: str
        value: int

    model = TestModel(name="test", value=42)
    request_with_model = {"data": model}
    key_with_model = cache.cache_key(request_with_model)
    assert isinstance(key_with_model, str)

    # Test with pydantic model class
    request_with_model_class = {"model_class": TestModel}
    key_with_model_class = cache.cache_key(request_with_model_class)
    assert isinstance(key_with_model_class, str)


def test_put_and_get(cache):
    """Test putting and getting from cache."""
    # Test putting and getting from memory cache
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini", "temperature": 0.7}

    value = DummyResponse(message="This is a test response", usage={"prompt_tokens": 10, "completion_tokens": 20})

    cache.put(request, value)
    result = cache.get(request)

    assert result.message == value.message
    assert result.usage == {}

    # Test with disk cache
    # First, clear memory cache to ensure we're using disk cache
    cache.reset_memory_cache()

    # Get from disk cache
    result_from_disk = cache.get(request)
    assert result_from_disk.message == value.message
    assert result_from_disk.usage == {}

    # Verify it was also added back to memory cache
    assert cache.cache_key(request) in cache.memory_cache


def test_cache_miss(cache):
    """Test getting a non-existent key."""
    request = {"prompt": "Non-existent", "model": "gpt-4"}
    result = cache.get(request)
    assert result is None


def test_cache_key_error_handling(cache):
    """Test error handling for unserializable objects."""

    # Test with a request that can't be serialized to JSON
    class UnserializableObject:
        pass

    request = {"data": UnserializableObject()}

    # Should not raise an exception
    result = cache.get(request)
    assert result is None

    # Should not raise an exception
    cache.put(request, "value")


def test_reset_memory_cache(cache):
    """Test resetting memory cache."""
    # Add some items to the memory cache
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    # Verify items are in memory cache
    for req in requests:
        key = cache.cache_key(req)
        assert key in cache.memory_cache

    # Reset memory cache
    cache.reset_memory_cache()

    # Verify memory cache is empty
    assert len(cache.memory_cache) == 0

    # But disk cache still has the items
    for req in requests:
        result = cache.get(req)
        assert result is not None


def test_save_and_load_memory_cache(cache, tmp_path):
    """Test saving and loading memory cache."""
    # Add some items to the memory cache
    requests = [{"prompt": f"Hello {i}", "model": "openai/gpt-4o-mini"} for i in range(5)]
    for i, req in enumerate(requests):
        cache.put(req, f"Response {i}")

    # Save memory cache to a temporary file
    temp_cache_file = tmp_path / "memory_cache.pkl"
    cache.save_memory_cache(str(temp_cache_file))

    # Create a new cache instance with disk cache disabled
    new_cache = Cache(
        enable_memory_cache=True,
        enable_disk_cache=False,
        disk_cache_dir=tmp_path / "disk_cache",
        disk_size_limit_bytes=0,
        memory_max_entries=100,
    )

    # Load the memory cache
    new_cache.load_memory_cache(str(temp_cache_file))

    # Verify items are in the new memory cache
    for req in requests:
        result = new_cache.get(req)
        assert result is not None
        assert result == f"Response {requests.index(req)}"


def test_request_cache_decorator(cache):
    """Test the lm_cache decorator."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache()
        def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        # Second call with same arguments should use cache
        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        # Call with different arguments should compute again
        result3 = test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


def test_request_cache_decorator_with_ignored_args_for_cache_key(cache):
    """Test the request_cache decorator with ignored_args_for_cache_key."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache(ignored_args_for_cache_key=["model"])
        def test_function1(prompt, model):
            return f"Response for {prompt} with {model}"

        @request_cache()
        def test_function2(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = test_function1(prompt="Hello", model="openai/gpt-4o-mini")
        result2 = test_function1(prompt="Hello", model="openai/gpt-4o")

        # Because model arg is ignored, the second call should return the same result as the first
        assert result1 == result2

        result3 = test_function2(prompt="Hello", model="openai/gpt-4o-mini")
        result4 = test_function2(prompt="Hello", model="openai/gpt-4o")

        # Because model arg is not ignored, the second call should return a different result
        assert result3 != result4


@pytest.mark.asyncio
async def test_request_cache_decorator_async(cache):
    """Test the request_cache decorator with async functions."""
    from dspy.clients.cache import request_cache

    # Mock the dspy.cache attribute
    with patch("dspy.cache", cache):
        # Define a test function
        @request_cache()
        async def test_function(prompt, model):
            return f"Response for {prompt} with {model}"

        # First call should compute the result
        result1 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
        assert result1 == "Response for Hello with openai/gpt-4o-mini"

        # Second call with same arguments should use cache
        with patch.object(cache, "get") as mock_get:
            mock_get.return_value = "Cached response"
            result2 = await test_function(prompt="Hello", model="openai/gpt-4o-mini")
            assert result2 == "Cached response"
            mock_get.assert_called_once()

        # Call with different arguments should compute again
        result3 = await test_function(prompt="Different", model="openai/gpt-4o-mini")
        assert result3 == "Response for Different with openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


def test_initialization_with_compression(tmp_path):
    """Test cache initialization with compression enabled."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
    )
    assert cache.compress == "gzip"
    assert isinstance(cache.disk_cache, FanoutCache)


def test_initialization_with_eviction_params(tmp_path):
    """Test cache initialization with eviction knobs."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        cull_limit=5,
        eviction_policy="least-recently-used",
    )
    assert isinstance(cache.disk_cache, FanoutCache)


def test_initialization_invalid_compression(tmp_path):
    """Test that invalid compression codec raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported compression codec"):
        Cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
            compress="lz4",
        )


def test_serialize_deserialize_gzip(tmp_path):
    """Test gzip compression round-trip via _serialize/_deserialize."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        compress="gzip",
    )
    original = {"key": "value", "nested": [1, 2, 3]}
    blob = cache._serialize(original)
    assert isinstance(blob, bytes)
    assert blob[:4] == _MAGIC_HEADER
    assert blob[4] == 0x01
    result = cache._deserialize(blob)
    assert result == original


def test_serialize_deserialize_zlib(tmp_path):
    """Test zlib compression round-trip via _serialize/_deserialize."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        compress="zlib",
    )
    original = {"key": "value", "nested": [1, 2, 3]}
    blob = cache._serialize(original)
    assert isinstance(blob, bytes)
    assert blob[:4] == _MAGIC_HEADER
    assert blob[4] == 0x02
    result = cache._deserialize(blob)
    assert result == original


def test_serialize_no_compression(tmp_path):
    """Test that no compression returns value unchanged."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        compress=None,
    )
    original = {"key": "value"}
    assert cache._serialize(original) is original


def test_put_and_get_with_gzip_compression(tmp_path):
    """Test full put/get cycle through disk with gzip compression."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
    )
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini"}
    value = DummyResponse(message="Compressed response", usage={"tokens": 10})

    cache.put(request, value)
    cache.reset_memory_cache()

    result = cache.get(request)
    assert result is not None
    assert result.message == "Compressed response"
    assert result.usage == {}


def test_put_and_get_with_zlib_compression(tmp_path):
    """Test full put/get cycle through disk with zlib compression."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="zlib",
    )
    request = {"prompt": "Hello", "model": "openai/gpt-4o-mini"}
    value = DummyResponse(message="Compressed response", usage={"tokens": 10})

    cache.put(request, value)
    cache.reset_memory_cache()

    result = cache.get(request)
    assert result is not None
    assert result.message == "Compressed response"


def test_backward_compatibility_legacy_entries(tmp_path):
    """Entries written without compression are still readable when compression is enabled."""
    cache_v1 = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress=None,
    )
    request = {"prompt": "Legacy", "model": "gpt-4"}
    cache_v1.put(request, "legacy_value")

    cache_v2 = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
    )
    result = cache_v2.get(request)
    assert result == "legacy_value"


def test_compression_reduces_disk_size(tmp_path):
    """Compressed entries should be smaller on disk than uncompressed ones."""
    dir_plain = tmp_path / "plain"
    dir_gzip = tmp_path / "gzip"

    cache_plain = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(dir_plain),
        compress=None,
    )
    cache_gzip = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(dir_gzip),
        compress="gzip",
    )

    large_value = {"text": "a" * 10000, "data": list(range(1000))}
    request = {"prompt": "size_test"}

    cache_plain.put(request, large_value)
    cache_gzip.put(request, large_value)

    plain_size = sum(f.stat().st_size for f in dir_plain.rglob("*") if f.is_file())
    gzip_size = sum(f.stat().st_size for f in dir_gzip.rglob("*") if f.is_file())

    assert gzip_size < plain_size


# ---------------------------------------------------------------------------
# TTL tests
# ---------------------------------------------------------------------------


def test_initialization_with_ttl(tmp_path):
    """TTL cache uses TTLCache for memory and stores ttl on the instance."""
    from cachetools import TTLCache

    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        ttl=60,
    )
    assert cache.ttl == 60
    assert isinstance(cache.memory_cache, TTLCache)


def test_initialization_without_ttl_uses_lru(tmp_path):
    """Without TTL the memory cache remains an LRUCache."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        ttl=None,
    )
    assert cache.ttl is None
    assert isinstance(cache.memory_cache, LRUCache)


def test_ttl_disk_entry_expires(tmp_path):
    """Disk entries written with a short TTL should expire."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        ttl=1,
    )
    request = {"prompt": "ephemeral"}
    cache.put(request, "short-lived")

    # Immediately readable
    assert cache.get(request) == "short-lived"

    # After sleeping past the TTL, entry should be gone
    import time

    time.sleep(1.5)
    assert cache.get(request) is None


def test_put_and_get_with_ttl(tmp_path):
    """Basic put/get works when TTL is set (entry still alive)."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        ttl=300,
    )
    request = {"prompt": "Hello", "model": "gpt-4"}
    value = DummyResponse(message="TTL response", usage={"tokens": 5})

    cache.put(request, value)
    cache.reset_memory_cache()

    result = cache.get(request)
    assert result is not None
    assert result.message == "TTL response"
    assert result.usage == {}


# ---------------------------------------------------------------------------
# Cross-feature integration tests (compression + TTL + eviction knobs)
# ---------------------------------------------------------------------------


def test_compression_and_ttl_combined_put_get(tmp_path):
    """Compressed entries with TTL are written and read back correctly."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
        ttl=300,
    )
    request = {"prompt": "combined", "model": "gpt-4"}
    value = DummyResponse(message="Compressed + TTL", usage={"tokens": 7})

    cache.put(request, value)

    # Verify from disk (bypass memory)
    cache.reset_memory_cache()
    result = cache.get(request)
    assert result is not None
    assert result.message == "Compressed + TTL"
    assert result.usage == {}


def test_compression_and_ttl_expiry(tmp_path):
    """Compressed entries still expire after the TTL elapses."""
    import time

    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="zlib",
        ttl=1,
    )
    request = {"prompt": "expire me"}
    cache.put(request, "temporary")

    assert cache.get(request) == "temporary"

    time.sleep(1.5)
    assert cache.get(request) is None


def test_zlib_compression_with_ttl_round_trip(tmp_path):
    """zlib + TTL round-trip preserves complex objects."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="zlib",
        ttl=300,
    )
    original = {"nested": {"a": [1, 2, 3]}, "text": "hello" * 100}
    request = {"prompt": "complex"}

    cache.put(request, original)
    result = cache.get(request)
    assert result == original


def test_all_knobs_combined(tmp_path):
    """compress + cull_limit + eviction_policy + ttl all set simultaneously."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
        cull_limit=5,
        eviction_policy="least-recently-used",
        ttl=300,
    )
    assert cache.compress == "gzip"
    assert cache.cull_limit == 5
    assert cache.eviction_policy == "least-recently-used"
    assert cache.ttl == 300

    request = {"prompt": "all knobs"}
    value = DummyResponse(message="Full config", usage={"tokens": 1})

    cache.put(request, value)
    cache.reset_memory_cache()

    result = cache.get(request)
    assert result is not None
    assert result.message == "Full config"


def test_compressed_entries_promote_to_memory_uncompressed(tmp_path):
    """Disk hit with compression should store the deserialized object in memory, not the blob."""
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
    )
    request = {"prompt": "promote"}
    value = {"data": "test"}

    cache.put(request, value)
    cache.reset_memory_cache()

    # Read from disk — should promote to memory
    result = cache.get(request)
    assert result == value

    key = cache.cache_key(request)
    mem_value = cache.memory_cache[key]
    # Memory should hold the dict, not compressed bytes
    assert isinstance(mem_value, dict)
    assert mem_value == value


def test_legacy_entry_readable_with_compression_and_ttl(tmp_path):
    """Legacy (no header) entries remain readable when both compression and TTL are enabled."""
    # Write with neither feature
    cache_v1 = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress=None,
        ttl=None,
    )
    request = {"prompt": "legacy"}
    cache_v1.put(request, "old_value")

    # Read with both features enabled
    cache_v2 = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
        ttl=300,
    )
    assert cache_v2.get(request) == "old_value"


def test_configure_cache_passes_all_params(tmp_path):
    """configure_cache() correctly plumbs compress, cull_limit, eviction_policy, and ttl."""
    from unittest.mock import patch as _patch

    import dspy
    from dspy.clients import configure_cache

    with _patch.dict(os.environ, {"DSPY_CACHEDIR": str(tmp_path)}):
        configure_cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=50,
            compress="zlib",
            cull_limit=3,
            eviction_policy="least-recently-used",
            ttl=120,
        )
        c = dspy.cache
        assert c.compress == "zlib"
        assert c.cull_limit == 3
        assert c.eviction_policy == "least-recently-used"
        assert c.ttl == 120


def test_set_ttl_preserves_compression(tmp_path):
    """set_ttl() rebuilds the cache but keeps the compression setting."""
    from unittest.mock import patch as _patch

    import dspy
    from dspy.clients import configure_cache, set_ttl

    with _patch.dict(os.environ, {"DSPY_CACHEDIR": str(tmp_path)}):
        configure_cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=str(tmp_path),
            compress="gzip",
            ttl=None,
        )
        assert dspy.cache.compress == "gzip"
        assert dspy.cache.ttl is None

        set_ttl(60)
        assert dspy.cache.ttl == 60
        assert dspy.cache.compress == "gzip"

        set_ttl(None)
        assert dspy.cache.ttl is None
        assert dspy.cache.compress == "gzip"


def test_switch_codec_reads_old_entries(tmp_path):
    """Entries written with gzip are still readable after switching to zlib."""
    cache_gzip = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="gzip",
    )
    request = {"prompt": "codec-switch"}
    cache_gzip.put(request, "gzip_value")

    cache_zlib = Cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=1024 * 1024,
        memory_max_entries=100,
        compress="zlib",
    )
    # _deserialize inspects the codec byte, not self.compress, so this should work
    assert cache_zlib.get(request) == "gzip_value"
