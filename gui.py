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
    
    def update_board(self,board):
        
        for i in range(8):
            for j in range(8):
                if board[i][j] == 1:
                    self.canvas.create_oval(90*i+5,90*j+5,90*(i+1)-5,90*(j+1)-5,fill="black")
                elif board[i][j] == -1:
                    self.canvas.create_oval(90*i+5,90*j+5,90*(i+1)-5,90*(j+1)-5,fill="white")
        self.update()
    def add_click_event(self,func):
        self.canvas.bind("<ButtonPress>",func)  
        

# def func(args):
#     print(args.x,args.y)

# gui = othello_gui()
# gui.add_click_event(func)
# gui.mainloop()
