import tkinter as tk
from tkinter import ttk
class game_mode:
    TITLE = 0
    GAME = 1
    GAMEOVER = 2
class othello_gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("オセロゲーム")
        self.geometry("720x720")
        self.id_list = []
        self.game_mode=game_mode.TITLE
        self.draw_background()
    def draw_background(self):
        
        self.canvas = tk.Canvas(
            self,
            width=720+1,
            height=720+1,
            bg="green"
            )
        self.canvas.pack()
        for i in range(8):
            for j in range(8):
                self.canvas.create_rectangle(90*i,90*j,90*(i+1),90*(j+1))
    def reset_canvas(self):
        for id in self.id_list:
            self.canvas.delete(id)
        self.id_list = []
    def update_board(self,board=None,score=None):
        if self.game_mode == game_mode.TITLE:
            print("TITLE")

            id = self.canvas.create_rectangle(130,330,235,400,fill="black")
            self.id_list.append(id)
            id = self.canvas.create_text(180, 360, text="先手", font=("Helvetica 20 bold"),fill="white")
            self.id_list.append(id)
            id = self.canvas.create_rectangle(590,330,485,400,fill="white")
            self.id_list.append(id)
            id = self.canvas.create_text(540, 360, text="後手", font=("Helvetica 20 bold"))
            self.id_list.append(id)

            id = self.canvas.create_text(360,100,text="手番を選択してください",font=("Helvetica  20 bold"))
            self.id_list.append(id)
            
        elif self.game_mode == game_mode.GAME:
            print("GAME")
            for i in range(8):
                for j in range(8):
                    if board[i][j] == 1:
                        id = self.canvas.create_oval(90*i+5,90*j+5,90*(i+1)-5,90*(j+1)-5,fill="black")
                        self.id_list.append(id)
                    elif board[i][j] == -1:
                        id = self.canvas.create_oval(90*i+5,90*j+5,90*(i+1)-5,90*(j+1)-5,fill="white")
                        self.id_list.append(id)
        elif self.game_mode == game_mode.GAMEOVER:
            print("GAMEOVER")
            # 背景の白い箱を描画
            id = self.canvas.create_rectangle(100,320,635,470,fill="white")
            id = self.canvas.create_text(360,360,text="ゲーム終了",font=("Helvetica  20 bold"))
            id = self.canvas.create_text(360,400,text=f"スコア: {score[0]} vs {score[1]}",font=("Helvetica  20 bold"))
            self.id_list.append(id)
            
            id = self.canvas.create_text(360,440,text="アプリケーションを終了してください",font=("Helvetica  20 bold"))
        self.update()
    def add_click_event(self,func):
        self.canvas.bind("<ButtonPress>",func)  
        

# def func(args):
#     print(args.x,args.y)

# gui = othello_gui()
# gui.add_click_event(func)
# gui.mainloop()
