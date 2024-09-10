from flask import render_template,Flask,request
from game import othello_class
no_AI = True
if not no_AI:
    from modellib import miniResNet,load_model
    from game import minimax_search2
def init_model():
    
    if no_AI:return
    global model,AI_class
    model = load_model()
    AI_class = minimax_search2(model)
    
    

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_move",methods=["POST"])
def get_move():
    r = request.json
    board = r["board"]
    turn = r["turn"]
    game = othello_class(
        undo_flg=True
    )
    game.board = board
    game.old_board = board
    game.turn = turn
    if not no_AI:
        score,move = AI_class.search(game,my_turn = game.turn,depth=3)
        game.apply_move(*move)
    elif len(game.get_valid_moves())>0:
        game.apply_move(*game.get_valid_moves()[0])
    return {
        "board":game.board,
        "valid_moves":game.get_valid_moves()
        }

@app.route("/is_valid",methods=["POST"])
def is_valid():
    r = request.json
    board = r["board"]
    turn = r["turn"]
    move = tuple(r["move"])
    game = othello_class()
    game.board = board
    game.turn = turn
    flg = False
    print("valid_move:",game.get_valid_moves())
    print("move:",move)
    if move in game.get_valid_moves():
        game.apply_move(*move)
        flg = True
    for l in game.board:
        print(l)
    return {
                "valid":flg,
                "board":game.board
                
            }
    
def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=5000,help="port number")
    parser = parser.parse_args()
    return {
        "port":parser.port
    }
if __name__ == "__main__":
    args = parse_arg()
    init_model()
    app.run(port=args["port"])