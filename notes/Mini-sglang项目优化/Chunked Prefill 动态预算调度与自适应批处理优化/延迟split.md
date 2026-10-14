应当拆成两个阶段：**检查前只读遍历并记录匹配位置；检查通过后直接分裂，不再从根匹配一遍。**我已按这个方式修改代码。

这里检查的是**预计加锁后的容量**，实际加锁在检查通过后进行。

1. **检查前：一次只读遍历**

    [`probe_prefix()` (line 173)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/kvcache/radix_cache.py:173) 返回的 `RadixMatchProbe` 现在记录四个字段：

    ```
    cached_len             # 总匹配长度
    newly_protected_size   # 加锁会新增保护的 token 数
    node                   # 最后一个匹配节点
    matched_in_node        # 在最后节点中匹配了多少 token
    ```

    部分匹配时只记录位置并停止，**不调用 `split_at()`**。

2. **检查通过后：直接提交分裂**

    新增 [`commit_match()` (line 200)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/kvcache/radix_cache.py:200)：

    ```
    def commit_match(self, probe: MatchProbe) -> RadixCacheHandle:
        assert isinstance(probe, RadixMatchProbe)
        node = probe.node
    
        if not node.is_root() and probe.matched_in_node != node.length:
            node = node.split_at(probe.matched_in_node)
    
        return RadixCacheHandle(probe.cached_len, node)
    ```

    它直接使用记录的节点和分裂位置，不再比较 token。

3. **把两个阶段连起来**

    [`CacheManager.try_match_and_lock()` (line 40)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/cache.py:40) 完成整个流程：

    ```
    probe = self.probe_req(req)
    
    required_size = (
        req.input_len - probe.cached_len
        + req.output_len + reserved_size
    )
    available_after_lock = self.available_size - probe.newly_protected_size
    
    if required_size > available_after_lock:
        return None
    
    handle = self.prefix_cache.commit_match(probe)
    self.lock(handle)
    self.touch(handle)
    return handle
    ```

因此，准入路径已经去掉了第二次 `match_req()` 遍历。**`lock()` 和 `touch()` 仍会沿父节点向上遍历**，用于维护引用计数和时间戳，但不会重复匹配 token。

这依赖当前串行调度，探测到提交之间缓存状态保持不变。更新后的 **40 项测试通过**，包括验证每条匹配边只比较一次。
