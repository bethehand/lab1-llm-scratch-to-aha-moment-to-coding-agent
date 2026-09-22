"""GSM8K 判分器 —— 三个脚本共用，改一处三处都生效。"""
import re

NUM = re.compile(r"-?\d+\.?\d*")


def extract(text):
    """
    → (答案字符串或 None, 是否输出了 #### 格式)

    ① 优先取 '#### N' 后面第一个数 —— 这是我们在 prompt 里要求的格式
    ② 否则取【最后一个】数字 —— GSM8K 评测的标准做法，答案总在结尾
       ★ 取最后一个而不是第一个：模型会先复述题目里的数字再推理，
         取第一个必然抓到题干的数（这就是刚才那个 bug）
       ★ 也比"任意数字对上就算"抗作弊：堆一串数字蒙答案，只有最后一个算数
    """
    t = text.replace(",", "").replace("$", "").replace("*", "")
    if "####" in t:
        m = NUM.findall(t.split("####")[-1])
        if m:
            return m[0], True
    m = NUM.findall(t)
    return (m[-1] if m else None), False


def reward(text, gold):
    a, _ = extract(text)
    if a is None:
        return 0.0
    try:
        return 1.0 if abs(float(a) - float(gold)) < 1e-4 else 0.0
    except ValueError:
        return 0.0


def has_format(text):
    return extract(text)[1]
