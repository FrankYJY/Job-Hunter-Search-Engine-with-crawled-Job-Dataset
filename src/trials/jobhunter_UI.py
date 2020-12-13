from tkinter.simpledialog import *
from tkinter import *


def xz():
    winNew = Toplevel(root)
    winNew.geometry('320x240')
    winNew.title('职业查询系统')
    lb2 = Label(winNew, text='输入你要查询的内容')
    lb2.pack()
    inp1 = Entry(winNew)
    inp1.pack()
    opQuery = Button(winNew, text='查询', command=winNew.destroy)
    opQuery.pack()
    btClose = Button(winNew, text='关闭', command=winNew.destroy)
    btClose.pack()
    txt = Text(winNew)
    txt.pack()


# def query()
# 这里放query函数
# a = string(inp1.get())  a表示获取输入的查询内容
# s= query函数
# txt.insert(END, s) 显示运算结果
# inp1.delete(0, END) 清空输入

root = Tk()
root.title('职业查询系统')  # title()方法设置标题文字
root.geometry('240x240')
lb = Label(root, text='请登录', \
           bg='#d3fbfb', \
           fg='black', \
           font=('Arial', 32), \
           width=10, \
           height=2, \
           relief=SUNKEN)
lb.pack()
lbred = Label(root, text='账号', fg='black', relief=GROOVE)
lbred.pack()
e1 = Entry(root, show=None, font=('Arial', 14))
e1.pack()
lbgreen = Label(root, text="密码", fg="black", relief=GROOVE)
lbgreen.pack()
e2 = Entry(root, show=None, font=('Arial', 14))
e2.pack()
lb = Label(root, text='')
lb.pack()
btn = Button(root, text='登录', command=xz)
btn.pack()
root.mainloop()

