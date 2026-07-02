"""
分析引擎：jieba 关键词提取、注意力计算、记忆衰减评估、矛盾检测
"""
import re
import jieba
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import get_analysis_config
from .database import (
    get_conversations_ordered, get_all_keywords, get_keyword_pairs,
    get_all_project_tags,
)


# 初始化 jieba
jieba.initialize()

# 停用词表（常见无意义词汇）
STOP_WORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如果", "因为", "所以", "但是", "可以", "这个", "那个",
    "这里", "那里", "这样", "那样", "还是", "或者", "并且", "而且",
    "嗯", "啊", "吧", "呢", "哈", "哦", "嘛", "呀",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "about", "up", "out", "if",
    "now", "its", "it", "you", "we", "he", "she", "they",
])


def _is_valid_keyword(word: str, min_len: int = 2) -> bool:
    """检查是否为有效关键词"""
    if len(word) < min_len:
        return False
    if word in STOP_WORDS:
        return False
    if re.match(r'^[\d\.\-\+\@\#\$%\^&\*\(\)\[\]\{\}]+$', word):
        return False
    return bool(re.search(r'[\u4e00-\u9fff\w]', word))


def extract_keywords(text: str, top_k: int = 30) -> list[tuple[str, float]]:
    """
    使用 jieba 提取关键词
    返回 [(keyword, weight), ...]
    """
    if not text:
        return []

    # jieba 分词
    words = jieba.cut(text)

    # 过滤并统计词频
    word_freq = Counter()
    for w in words:
        w = w.strip().lower()
        if _is_valid_keyword(w):
            word_freq[w] += 1

    # 按频率排序取 Top-K
    top_words = word_freq.most_common(top_k)

    if not top_words:
        return []

    max_freq = top_words[0][1]
    return [(w, freq / max_freq) for w, freq in top_words]


def build_keyword_graph(limit: int = 50) -> dict:
    """
    构建关键词关联网络图数据
    返回 Plotly 力导向图所需的数据格式
    """
    cfg = get_analysis_config()
    top_k = cfg.get("keyword_top_k", 30)

    keywords_stats = get_all_keywords()
    pairs = get_keyword_pairs(min_cooccur=1)

    if not keywords_stats:
        return {"nodes": [], "edges": []}

    # 取 top-N 关键词作为节点
    top_keywords = keywords_stats[:top_k]
    node_names = set(k["keyword"] for k in top_keywords)

    # 构建节点
    max_freq = top_keywords[0]["freq"] if top_keywords else 1
    nodes = []
    for kw in top_keywords:
        nodes.append({
            "id": kw["keyword"],
            "label": kw["keyword"],
            "size": max(8, min(30, (kw["freq"] / max_freq) * 25)),
            "freq": kw["freq"],
        })

    # 构建连线（只保留两端都在 top-N 节点内的）
    edges = []
    for pair in pairs:
        if pair["source"] in node_names and pair["target"] in node_names:
            edges.append({
                "source": pair["source"],
                "target": pair["target"],
                "weight": pair["cooccur"],
            })

    return {"nodes": nodes, "edges": edges}


def calc_attention_scores(text: str, logprobs: list = None) -> list[dict]:
    """
    计算注意力分数
    - 有 logprobs 时，基于 logprobs 的 logprob 值转换
    - 无 logprobs 时，基于 jieba 词频作为伪注意力
    返回 [{"word": str, "score": float}, ...]
    """
    if not text:
        return []

    words = list(jieba.cut(text))
    words = [w.strip() for w in words if w.strip()]

    if logprobs and len(logprobs) > 0:
        # 有 logprobs 数据：取每个 token 的 logprob，softmax 归一化
        scores = []
        for item in logprobs:
            lp = item.get("logprob", 0)
            # logprob 通常是负数，exp 转为概率
            scores.append(np.exp(lp) if lp < 0 else lp)

        # 如果有 top_logprobs 信息，取 top-1 的 logprob
        # 这里用原始 token 级别的 logprob
        # 一般 logprobs 是针对 AI 输出的，这里做近似映射到用户输入词
        if len(scores) <= len(words):
            # 不够的词补 0
            scores = scores + [0.0] * (len(words) - len(scores))
        else:
            scores = scores[:len(words)]

        # 归一化到 0-1
        max_s = max(scores) if scores else 1
        if max_s > 0:
            scores = [s / max_s for s in scores]
    else:
        # 降级：基于词频 + 位置衰减的伪注意力（避免全部相同分数）
        word_freq = Counter(words)
        if word_freq:
            total = len(words)
            max_f = max(word_freq.values())
            scores = []
            for i, w in enumerate(words):
                # 词频因子 + 位置因子（后半段词加一点权重衰减，制造变化）
                freq_factor = word_freq[w] / max_f if max_f > 0 else 1.0
                pos_factor = 0.7 + 0.3 * (1 - i / total) if total > 0 else 1.0
                scores.append(freq_factor * pos_factor)
        else:
            scores = [0.5] * len(words)

    return [{"word": w, "score": round(s, 4)} for w, s in zip(words, scores)]


def calc_memory_decay(project_tag: str = "", window_size: int = 50) -> list[dict]:
    """
    计算记忆衰减曲线
    返回 [{"index": int, "accuracy": float, "timestamp": str, "preview": str}, ...]
    """
    cfg = get_analysis_config()
    window_size = cfg.get("decay_window", window_size)

    conversations = get_conversations_ordered(project_tag, limit=window_size)

    if len(conversations) < 3:
        return [{
            "index": i,
            "accuracy": 1.0,
            "timestamp": c.get("timestamp", ""),
            "preview": (c.get("user_input", "") or "")[:30]
        } for i, c in enumerate(conversations)]

    # 提取所有用户输入文本
    texts = []
    for c in conversations:
        user_input = c.get("user_input", "") or ""
        ai_output = c.get("ai_output", "") or ""
        combined = f"{user_input} {ai_output}"
        texts.append(combined)

    # 使用 TF-IDF 计算每条对话与前面所有对话的语义相似度
    # sklearn 1.9+: 先 jieba 分词再空格拼接，兼容新版 API
    tokenized_texts = [" ".join(jieba.cut(t)) for t in texts]
    vectorizer = TfidfVectorizer(max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
    except ValueError:
        # 如果所有文本都为空
        return [{
            "index": i,
            "accuracy": 1.0,
            "timestamp": c.get("timestamp", ""),
            "preview": (c.get("user_input", "") or "")[:30]
        } for i, c in enumerate(conversations)]

    results = []
    for i, c in enumerate(conversations):
        if i == 0:
            accuracy = 1.0
        else:
            # 计算当前对话与历史对话的平均相似度
            current_vec = tfidf_matrix[i:i+1]
            history_vecs = tfidf_matrix[:i]
            similarities = cosine_similarity(current_vec, history_vecs)[0]
            accuracy = float(np.mean(similarities)) if len(similarities) > 0 else 1.0

            # 加权：越近的对话权重越高
            if len(similarities) > 1:
                weights = np.linspace(0.5, 1.0, len(similarities))
                accuracy = float(np.average(similarities, weights=weights))

        results.append({
            "index": i,
            "accuracy": round(accuracy, 4),
            "timestamp": c.get("timestamp", ""),
            "preview": (c.get("user_input", "") or "")[:30]
        })

    return results


def detect_contradictions(recent_n: int = 20) -> list[dict]:
    """
    检测矛盾与潜在幻觉
    返回 [{"type": str, "description": str, "severity": str, "conv_id": int}, ...]
    """
    cfg = get_analysis_config()
    threshold = cfg.get("contradiction_threshold", 0.75)

    conversations = get_conversations_ordered(limit=recent_n)

    if len(conversations) < 2:
        return []

    alerts = []
    texts = []
    for c in conversations:
        ai_output = c.get("ai_output", "") or ""
        texts.append(ai_output)

    # 检测逻辑矛盾：相邻回答之间相似度过低可能是矛盾
    # 检测回答中引用"之前提到"但实际上没有的幻觉
    reference_pattern = re.compile(
        r'(之前|前面|刚刚|上面|刚才|上次|之前|之前说|说过|提到).{0,10}(说|讲|提到|讨论)'
    )

    for i, c in enumerate(conversations):
        ai_output = c.get("ai_output", "") or ""

        # 检测引用幻觉
        ref_match = reference_pattern.search(ai_output)
        if ref_match and i > 0:
            # 检查前一条对话是否有关联内容
            prev_output = texts[i - 1] if i > 0 else ""
            if prev_output:
                # 简单启发式：如果当前回答引用了之前的讨论但两者 TF-IDF 相似度过低
                try:
                    tokenized = [" ".join(jieba.cut(prev_output[-200:])),
                                 " ".join(jieba.cut(ai_output[:200]))]
                    vectorizer = TfidfVectorizer(max_features=100)
                    vecs = vectorizer.fit_transform(tokenized)
                    sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
                    if sim < 0.1:
                        alerts.append({
                            "type": "reference_hallucination",
                            "description": f"AI 引用了之前的讨论，但当前回答与上文关联度极低 (相似度: {sim:.2f})，可能是幻觉引用。",
                            "severity": "danger",
                            "conv_id": c["id"],
                        })
                except ValueError:
                    pass

    # 检测记忆断崖：衰减曲线连续下降
    decay_results = calc_memory_decay(recent_n)
    danger_threshold = cfg.get("memory_danger_threshold", 0.3)

    consecutive_low = 0
    for d in decay_results[-10:]:
        if d["accuracy"] < danger_threshold:
            consecutive_low += 1
        else:
            consecutive_low = 0

        if consecutive_low >= 3:
            alerts.append({
                "type": "memory_danger",
                "description": f"检测到记忆衰减连续低于阈值，AI 可能正在遗忘早期需求，幻觉风险上升。",
                "severity": "warning",
                "conv_id": None,
            })
            break

    return alerts
