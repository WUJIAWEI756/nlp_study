import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

from openai import OpenAI
from transformers import AutoTokenizer

# 读取数据
ratings = pd.read_csv("./M_ML-100k/ratings.dat", sep="::", header=None, engine="python")
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100k/movies.dat", sep="::", header=None, engine="python", encoding='latin-1')
movies.columns = ["movie_id", "movie_title", "movie_tag"]

# 配置参数
config = {
    'data_path': './M_ML-100k',
    'max_len': 10,  
    'batch_size': 32,
    'num_epochs': 5,
    'llm_model': 'qwen-max',
    'top_k': 10,
    'test_split': 0.1,
    'validation_split': 0.1,
    'temperature': 0.7,
    'max_tokens': 500,
}


class GPT4RecDataset:
    """
    数据处理类，与BERT4Rec保持一致的数据处理方式
    """

    def __init__(self, config):
        self.config = config
        self.df = ratings

        # 过滤评分，保留3分及以上作为正样本
        self.df = self.df[self.df['rating'] >= 3].reset_index(drop=True)

        # 创建编码器
        self.item_encoder, self.item_decoder = self._create_encoder_decoder('movie_id')
        self.user_encoder, self.user_decoder = self._create_encoder_decoder('user_id')

        # 创建电影名称映射
        self.movie_id_to_name = dict(zip(movies['movie_id'], movies['movie_title']))
        self.movie_id_to_tags = dict(zip(movies['movie_id'], movies['movie_tag']))

        # 编码数据
        self.df['item_idx'] = self.df['movie_id'].apply(lambda x: self.item_encoder[x])
        self.df['user_idx'] = self.df['user_id'].apply(lambda x: self.user_encoder[x])

        # 按用户和时间排序
        self.df = self.df.sort_values(['user_idx', 'timestamp'])

        # 划分数据集
        self.user_train, self.user_valid, self.user_test = self._split_data()

        print(f"用户数量: {len(self.user_encoder)}")
        print(f"电影数量: {len(self.item_encoder)}")
        print(f"训练用户数: {len(self.user_train)}")
        print(f"验证用户数: {len(self.user_valid)}")
        print(f"测试用户数: {len(self.user_test)}")

    def _create_encoder_decoder(self, col: str):
        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(sorted(ids)):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def _split_data(self):
        """按时间划分用户的序列"""
        user_sequences = defaultdict(list)

        # 按用户分组
        for user_idx, group in self.df.groupby('user_idx'):
            items = group['item_idx'].tolist()
            user_sequences[user_idx] = items

        user_train, user_valid, user_test = {}, {}, {}

        for user, items in user_sequences.items():
            seq_len = len(items)

            if seq_len >= 3:
                user_test[user] = [items[-1]]  # 最后一个作为测试
                user_valid[user] = [items[-2]]  # 倒数第二个作为验证
                user_train[user] = items[:-2]  # 其余作为训练
            elif seq_len == 2:
                user_test[user] = [items[-1]]
                user_valid[user] = [items[-2]]
                user_train[user] = []
            elif seq_len == 1:
                user_test[user] = [items[0]]
                user_valid[user] = []
                user_train[user] = []

        # 过滤训练序列太短的用户
        valid_users = [user for user in user_train if len(user_train[user]) >= 1]
        user_train = {user: user_train[user] for user in valid_users}
        user_valid = {user: user_valid[user] for user in valid_users if user in user_valid}
        user_test = {user: user_test[user] for user in valid_users if user in user_test}

        return user_train, user_valid, user_test

    def get_movie_info(self, movie_idx: int) -> str:
        """获取电影信息"""
        movie_id = self.item_decoder.get(movie_idx)
        if movie_id:
            title = self.movie_id_to_name.get(movie_id, f"Movie_{movie_id}")
            tags = self.movie_id_to_tags.get(movie_id, "")
            return f"{title} ({tags})" if tags else title
        return f"Movie_{movie_idx}"

    def get_user_history_text(self, user_idx: int, max_items: int = 10) -> str:
        """获取用户历史记录文本"""
        if user_idx not in self.user_train:
            return ""

        items = self.user_train[user_idx][-max_items:]  # 取最近的历史
        movie_texts = []
        for item_idx in items:
            movie_text = self.get_movie_info(item_idx)
            movie_texts.append(movie_text)

        return "\n".join(movie_texts)


class GPT4Rec:
    """
    GPT4Rec推荐系统
    """

    def __init__(self, config, api_key=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化数据集
        self.dataset = GPT4RecDataset(config)

        # 初始化LLM客户端
        if api_key is None:
            api_key = os.getenv('QWEN_API_KEY', 'your-api-key-here')

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )

        # 缓存推荐结果以减少API调用
        self.recommendation_cache = {}

    def build_prompt(self, user_history: str) -> str:
        """构建推荐提示"""
        prompt = f"""你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影。

用户历史观看的电影：
{user_history}

请基于上述观看历史，推荐10部用户可能会喜欢的电影。
请严格按照以下格式输出：
1. 电影名称1
2. 电影名称2
...
10. 电影名称10

只输出电影名称列表，不要有其他文字。"""
        return prompt

    def call_llm_api(self, prompt: str) -> str:
        """调用Qwen API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config['llm_model'],
                messages=[
                    {"role": "system", "content": "你是一个专业的电影推荐系统。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用错误: {e}")
            return ""

    def parse_recommendations(self, response: str) -> List[str]:
        """解析LLM返回的推荐结果"""
        recommendations = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # 匹配数字开头的行，如"1. 电影名称"或"1) 电影名称"
            if line and (line[0].isdigit() or '.' in line[:3]):
                # 移除数字和标点
                movie_name = line.split('.', 1)[-1].strip()
                movie_name = movie_name.split(')', 1)[-1].strip()
                if movie_name:
                    recommendations.append(movie_name)

        return recommendations[:self.config['top_k']]

    def get_movie_id_by_name(self, movie_name: str) -> int:
        """根据电影名称查找电影ID"""
        # 尝试精确匹配
        for movie_id, title in self.dataset.movie_id_to_name.items():
            if movie_name.lower() == title.lower():
                return self.dataset.item_encoder.get(movie_id, -1)

        # 尝试部分匹配
        for movie_id, title in self.dataset.movie_id_to_name.items():
            if movie_name.lower() in title.lower() or title.lower() in movie_name.lower():
                return self.dataset.item_encoder.get(movie_id, -1)

        # 尝试匹配电影ID
        try:
            if movie_name.startswith("Movie_"):
                movie_num = int(movie_name.split("_")[1])
                return movie_num
        except:
            pass

        return -1

    def recommend_for_user(self, user_idx: int, use_cache: bool = True) -> List[int]:
        """为用户生成推荐"""
        if use_cache and user_idx in self.recommendation_cache:
            return self.recommendation_cache[user_idx]

        # 获取用户历史
        user_history = self.dataset.get_user_history_text(user_idx)

        if not user_history:
            # 如果用户没有历史记录，返回空列表
            return []

        # 构建提示并调用API
        prompt = self.build_prompt(user_history)
        response = self.call_llm_api(prompt)

        if not response:
            # 如果API调用失败，返回随机推荐作为fallback
            return self._fallback_recommendation(user_idx)

        # 解析推荐结果
        movie_names = self.parse_recommendations(response)

        # 转换为电影ID
        recommendations = []
        for movie_name in movie_names:
            movie_id = self.get_movie_id_by_name(movie_name)
            if movie_id != -1 and movie_id not in recommendations:
                recommendations.append(movie_id)

        # 如果推荐数量不足，用随机推荐补充
        if len(recommendations) < self.config['top_k']:
            recommendations.extend(self._get_random_recommendations(
                user_idx,
                self.config['top_k'] - len(recommendations),
                exclude=recommendations
            ))

        # 缓存结果
        self.recommendation_cache[user_idx] = recommendations[:self.config['top_k']]

        return recommendations[:self.config['top_k']]

    def _fallback_recommendation(self, user_idx: int) -> List[int]:
        """回退推荐策略"""
        # 获取用户历史看过的电影
        if user_idx in self.dataset.user_train:
            watched = set(self.dataset.user_train[user_idx])
        else:
            watched = set()

        # 获取所有电影ID
        all_movies = list(range(len(self.dataset.item_encoder)))

        # 选择用户没看过的电影
        candidates = [m for m in all_movies if m not in watched]

        # 随机选择
        if len(candidates) >= self.config['top_k']:
            return list(np.random.choice(candidates, self.config['top_k'], replace=False))
        else:
            return candidates + list(np.random.choice(all_movies,
                                                      self.config['top_k'] - len(candidates),
                                                      replace=False))

    def _get_random_recommendations(self, user_idx: int, n: int, exclude: List[int] = None) -> List[int]:
        """获取随机推荐"""
        if exclude is None:
            exclude = []

        # 获取用户历史
        if user_idx in self.dataset.user_train:
            watched = set(self.dataset.user_train[user_idx])
        else:
            watched = set()

        # 排除已推荐和已观看的
        all_movies = list(range(len(self.dataset.item_encoder)))
        candidates = [m for m in all_movies if m not in watched and m not in exclude]

        if len(candidates) >= n:
            return list(np.random.choice(candidates, n, replace=False))
        else:
            return candidates

    def evaluate(self, user_data: Dict[int, List[int]], split: str = "test") -> Tuple[float, float]:
        """评估模型性能"""
        hr_scores = []
        ndcg_scores = []

        print(f"\n评估{split.upper()}集...")

        for user_idx, true_items in tqdm(user_data.items(), desc=f"评估{split}"):
            if not true_items:
                continue

            # 获取推荐结果
            recommendations = self.recommend_for_user(user_idx)

            # 计算HR@K
            hit = 0
            for true_item in true_items:
                if true_item in recommendations:
                    hit = 1
                    # 计算NDCG
                    rank = recommendations.index(true_item) + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                    ndcg_scores.append(ndcg)
                    break

            hr_scores.append(hit)

        if not hr_scores:
            return 0.0, 0.0

        hr = np.mean(hr_scores)
        ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

        return hr, ndcg

    def train(self):
        """GPT4Rec不需要传统训练，这里实现一个模拟训练过程"""
        print("GPT4Rec使用预训练模型，无需传统训练...")
        print("开始验证集评估...")

        # 在验证集上评估
        val_hr, val_ndcg = self.evaluate(self.dataset.user_valid, "validation")
        print(f"验证集 - HR@{self.config['top_k']}: {val_hr:.4f}, NDCG@{self.config['top_k']}: {val_ndcg:.4f}")

    def test(self):
        """在测试集上评估"""
        print("\n开始测试集评估...")
        test_hr, test_ndcg = self.evaluate(self.dataset.user_test, "test")
        print(f"测试集 - HR@{self.config['top_k']}: {test_hr:.4f}, NDCG@{self.config['top_k']}: {test_ndcg:.4f}")

    def generate_example_recommendation(self, user_idx: int = None):
        """生成示例推荐"""
        if user_idx is None:
            # 选择一个有足够历史的用户
            valid_users = [u for u in self.dataset.user_train.keys()
                           if len(self.dataset.user_train[u]) >= 3]
            if valid_users:
                user_idx = valid_users[0]
            else:
                print("没有找到合适的用户")
                return

        print(f"\n为用户 {user_idx} 生成推荐:")
        print("-" * 50)

        # 获取用户历史
        user_history = self.dataset.get_user_history_text(user_idx)
        print("用户历史观看的电影:")
        print(user_history)
        print("-" * 50)

        # 获取推荐
        recommendations = self.recommend_for_user(user_idx)

        print(f"\nTop-{self.config['top_k']} 推荐:")
        for i, movie_idx in enumerate(recommendations):
            movie_name = self.dataset.get_movie_info(movie_idx)
            print(f"{i + 1}. {movie_name}")

        # 显示真实的下一个物品
        if user_idx in self.dataset.user_test:
            true_next = self.dataset.user_test[user_idx][0]
            true_movie_name = self.dataset.get_movie_info(true_next)
            print(f"\n真实的下一个观看: {true_movie_name}")

            # 检查是否命中
            if true_next in recommendations:
                rank = recommendations.index(true_next) + 1
                print(f"✓ 推荐命中! 排名第 {rank}")
                print(f"NDCG贡献: {1.0 / np.log2(rank + 1):.4f}")
            else:
                print("✗ 未命中推荐")


def main():
    # 初始化GPT4Rec
    print("初始化GPT4Rec...")
    gpt4rec = GPT4Rec(config)

    # 训练（实际上只是评估验证集）
    gpt4rec.train()

    # 测试
    gpt4rec.test()

    # 生成示例推荐
    print("\n" + "=" * 50)
    print("示例推荐生成")
    print("=" * 50)
    gpt4rec.generate_example_recommendation()


if __name__ == "__main__":
    api_key = os.getenv('QWEN_API_KEY')
    if not api_key:
        print("警告: 未设置QWEN_API_KEY环境变量")
        print("export QWEN_API_KEY='your-api-key'")
        print("\n" + "=" * 50)

    main()
