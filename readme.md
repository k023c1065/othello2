# 概要
フレームワークとしてTensorflowを利用したAIオセロです。

# セットアップ

1. (任意)venv等の仮想環境をセットアップし、アクティベートする
   ```bash
   python3 -m venv /(新しい仮想環境のパス)
   source /(新しい仮想環境のパス)/bin/activate #(For linux)
   /(新しい仮想環境のパス)/Scripts/activate.bat #(For windows)
   ```

2. 必要なライブラリをインストールする。
   ```bash
   python3 -m pip install -r requirements.txt
   ```

# プレイ方法

1. 以下のコマンドでゲームを開始する。
   ```bash
   python3 main.py
   ```
2. ゲームが開始する。
    ```bash
    Starting the game...
      0 1 2 3 4 5 6 7
    0 □ □ □ □ □ □ □ □
    1 □ □ □ □ □ □ □ □
    2 □ □ □ □ □ □ □ □
    3 □ □ □ ● ○ □ □ □
    4 □ □ □ ○ ● □ □ □
    5 □ □ □ □ □ □ □ □
    6 □ □ □ □ □ □ □ □
    7 □ □ □ □ □ □ □ □
    turn: 1 symbol: ●
    valid moves:[(2, 4), (3, 5), (4, 2), (5, 3)]
    AI is thinking...Done. Took 1.08sec(s)
    Applying move... 2 , 4
      0 1 2 3 4 5 6 7
    0 □ □ □ □ □ □ □ □
    1 □ □ □ □ □ □ □ □
    2 □ □ □ □ ● □ □ □
    3 □ □ □ ● ● □ □ □
    4 □ □ □ ○ ● □ □ □
    5 □ □ □ □ □ □ □ □
    6 □ □ □ □ □ □ □ □
    7 □ □ □ □ □ □ □ □
    turn: -1 symbol: ○
    valid moves:[(2, 3), (2, 5), (4, 5)]
    Your turn.Enter your next move:
    ```
    上記のような入力待機画面になったら自身の次の手を打ちます。
    $i$行$j$列に自身の石を置きたい場合は$i\ j$という風に間にスペースを入れるように入力してください。
    ##### エラー例
    ```bash
    [ERROR]Invalid move. Please enter a valid move.
    ```
    石を置くことができない場所に置こうとしました。実際に置くことができる場所はvalid movesの隣に書いてあるため参考にしてください。
    ```bash
    [ERROR]Unexpected input. Please enter two integers separated by a space.
    ```
    入力が不正です。$i$と$j$の間にはスペースを入れてください。

3. ゲームが終了したらスコアが計算され勝者がチャットに表示されます。
   ```bash
    Score: 44 vs 20
    Result: Player 1 wins!
    Press Enter to continue...
    ```
    Enterキーを押すと新しい試合を始めます。
    終了したい場合はCtrl+Cを押してください。

# モデルのダウンロード方法
デフォルトで一つ入れています。
wip
