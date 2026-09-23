from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
'''
负责管理 GPU 上 KV Cache 的物理 Block：
哪些 Block 空闲、哪些被使用、哪些可以复用，
以及哪些 token 前缀已经缓存到 KV Cache 中。

每个 Block 的大小是固定的，BlockManager 
负责将 Sequence 中的 token 划分为若干个 Block，

并管理这些 Block 的分配和释放。
'''

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0      #表示当前有多少个序列（Sequence）正在共享这个块
        self.hash = -1      #这个块内容的链式哈希值
        self.token_ids = []     #这个块实际存储的 token id 列表，长度通常等于 block_size（满块）

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]    #创建 num_blocks 个 Block 对象，每个块的 block_id 就是它的索引 i
        self.hash_to_block_id: dict[int, int] = dict()  #键是块内容的链式哈希值，值是对应的物理块 ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #当前没有被使用的 Block
        self.used_block_ids: set[int] = set()   #当前正在使用的 Block

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        '''
        为一段token_ids计算链式哈希值，prefix是前一个块的哈希值
        只有哈希相同且 token_ids 相同的块，才能被多个序列共享
        '''
        h = xxhash.xxh64()
        if prefix != -1:    #如果 prefix != -1，说明当前块不是第一个块，需要把前一个块的哈希拼进来
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes()) #加入当前块的 token 数据
        return h.intdigest()

    def _allocate_block(self) -> int:
        '''
        从空闲队列中取出一个物理块，重置它的状态，并标记为已使用
        '''
        block_id = self.free_block_ids.popleft()    #从 free_block_ids（空闲块 ID 的双端队列）的左端弹出一个块 ID
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        '''
        在真正给某个 Sequence 分配 KV Cache 块之前，
        先判断当前空闲块是否足够，
        并顺便计算出这个序列有多少个前缀块可以复用缓存
        '''
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        '''
        真正为一个 Sequence 分配 KV Cache 物理块。
        它会先复用已缓存的前缀块，再为剩余部分分配新块，
        并把分配结果记录到 seq.block_table 中
        '''
        assert not seq.block_table      #
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        '''
        释放一个序列所占用的所有 KV Cache 块
        '''
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        #负责预检查是否有足够空闲块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
       # 负责在需要时实际分配新块
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        '''将已经填满的 KV Cache 块注册到前缀哈希索引的方法。
        它的作用是在 prefill 阶段完成后，
        把新生成的满块内容（token_ids 和链式哈希）记录到 hash_to_block_id 中，
        这样后续其他序列如果有相同前缀，就能通过哈希找到并复用这些块，避免重复计算和存储。
        '''
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
