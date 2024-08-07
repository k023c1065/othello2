import multiprocessing as mp
import random
import time
import numpy as np

def child_process(pipe):
    while True:
        # データの生成
        array = np.random.rand(3, 3)
        print(f"Child process {mp.current_process().name} generated array:\n{array}")
        pipe.send(array)  # 生成した配列を親プロセスに送信

        # 親プロセスからの受信
        processed_array = pipe.recv()
        if processed_array == 'END':
            break
        print(f"Child process {mp.current_process().name} received processed array:\n{processed_array.sum()}")
    
    pipe.close()

def process_array(array):
    # 親プロセスでの処理（ここでは配列の各要素を2倍にする）
    return array * 2

def main():
    proc_num = 16 # プロセスの数
    processes = []
    pipes = []

    # 子プロセスの生成と通信のためのパイプの設定
    for _ in range(proc_num):
        parent_pipe, child_pipe = mp.Pipe()
        p = mp.Process(target=child_process, args=(child_pipe,))
        processes.append(p)
        pipes.append(parent_pipe)

    # 子プロセスを開始
    for p in processes:
        p.start()

    # 親プロセスでの受信と処理
    while True:
        for parent_pipe in pipes:
            received_array = parent_pipe.recv()
            print(f"Parent process received array:\n{received_array}")

            processed_array = process_array(received_array)  # 配列を処理
            parent_pipe.send(processed_array)  # 処理した配列を子プロセスに送信

    # 終了メッセージを子プロセスに送信
    for parent_pipe in pipes:
        parent_pipe.send('END')

    # 子プロセスの終了待機
    for p in processes:
        p.join()
def wait(i):
    print("Start")
    print("Finished")
    return random.randint(0,100)
if __name__ == "__main__":
    from transformers import pipeline

    # 感情分析のためのパイプラインを作成
    classifier = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

    # 分析するテキスト
    texts = [
    "I love this product! It works great.",
    "This is the worst experience I've ever had.",
    "I'm feeling quite neutral about this.",
    "It's okay, not great but not bad either."
    ]

    # 感情分析を実行
    results = classifier(texts)

    # 結果を表示
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")