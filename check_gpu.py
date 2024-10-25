import torch
import subprocess

def get_free_gpu():
    try:
        # nvidia-smiの出力を取得
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        # 出力をリストに変換
        free_memories = [int(x) for x in result.strip().split('\n')]
        
        # 最大のメモリを持つGPUのインデックスを取得
        max_memory_gpu = free_memories.index(max(free_memories))
        return max_memory_gpu
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None
