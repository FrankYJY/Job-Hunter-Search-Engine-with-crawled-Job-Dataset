import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import pickle
from PIL import Image,ImageTk

from src.data_dealer import *

'''
RUN THIS FILE
'''

'''global'''
record_list = []
text_list = []
ranked_similarity = []
first_time_main=True
current_page = 0
page_n = None
v = None

'''UI'''
# 窗口
# window
window = tk.Tk()
window.title('Welcome to Job Hunter Search Engine')
window.geometry('450x300')
# 画布放置图片
# put image
canvas = tk.Canvas(window, height=300, width=500)
imagefile = tk.PhotoImage(file='background.png')
image = canvas.create_image(0, 0, anchor='nw', image=imagefile)
canvas.pack(side='top')
#标题
# title
title = tk.Label(window, text = 'Job Hunter', font=("微软雅黑", 24)).place(x=150, y=60)
# 标签 用户名密码
# label: username and password
tk.Label(window, text='User Name:').place(x=100, y=150)
tk.Label(window, text='Password:').place(x=100, y=190)
# 用户名输入框
# username input box
var_usr_name = tk.StringVar()
entry_usr_name = tk.Entry(window, textvariable=var_usr_name)
entry_usr_name.place(x=180, y=150)
# 密码输入框
# password input box
var_usr_pwd = tk.StringVar()
entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*')
entry_usr_pwd.place(x=180, y=190)


# 登录函数
def usr_log_in():
    # get user input
    usr_name = var_usr_name.get()
    usr_pwd = var_usr_pwd.get()
    # 从本地字典获取用户信息，如果没有则新建本地数据库
    # try to get user database, or new one
    try:
        with open('usr_info.pickle', 'rb') as usr_file:
            usrs_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usr_info.pickle', 'wb') as usr_file:
            usrs_info = {'admin': 'admin'}
            pickle.dump(usrs_info, usr_file)
    # 判断用户名和密码是否匹配
    # validate
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            tk.messagebox.showinfo(title='welcome',
                                   message='login success: ' + usr_name)
            window.destroy()
            query_in()
        else:
            tk.messagebox.showerror(message='password wrong')
    elif usr_name == '' or usr_pwd == '':
        tk.messagebox.showerror(message='username or password is empty')
    else:
        is_signup = tk.messagebox.askyesno('Welcome!', "You haven't register yet, do you want to register now?")
        if is_signup:
            usr_sign_up()

# 注册函数
def usr_sign_up():
    # 确认注册时的相应函数
    def signtowcg():
        # 获取输入框内的内容
        nn = new_name.get()
        np = new_pwd.get()
        npf = new_pwd_confirm.get()

        # 本地加载已有用户信息,如果没有则已有用户信息为空
        try:
            with open('usr_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)
        except FileNotFoundError:
            exist_usr_info = {}

            # 检查用户名存在、密码为空、密码前后不一致
        if nn in exist_usr_info:
            tk.messagebox.showerror('wrong', 'username already exist')
        elif np == '' or nn == '':
            tk.messagebox.showerror('wrong', 'username or password is empty')
        elif np != npf:
            tk.messagebox.showerror('wrong', 'passwords are different')
        # 注册信息没有问题则将用户名密码写入数据库
        else:
            exist_usr_info[nn] = np
            with open('usr_info.pickle', 'wb') as usr_file:
                pickle.dump(exist_usr_info, usr_file)
            tk.messagebox.showinfo('welcome', 'register success')
            # 注册成功关闭注册框
            window_sign_up.destroy()

    # 新建注册界面
    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('register')
    # 用户名变量及标签、输入框
    new_name = tk.StringVar()
    tk.Label(window_sign_up, text='username:').place(x=10, y=10)
    tk.Entry(window_sign_up, textvariable=new_name).place(x=150, y=10)
    # 密码变量及标签、输入框
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text='enter password:').place(x=10, y=50)
    tk.Entry(window_sign_up, textvariable=new_pwd, show='*').place(x=150, y=50)
    # 重复密码变量及标签、输入框
    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up, text='enter password again：').place(x=10, y=90)
    tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*').place(x=150, y=90)
    # 确认注册按钮及位置
    bt_confirm_sign_up = tk.Button(window_sign_up, text='confirm registration',
                                   command=signtowcg)
    bt_confirm_sign_up.place(x=150, y=130)

# 搜索主界面
# main window
def query_in():
    def search():
        '''simple search'''
        global record_list
        global text_list
        global ranked_similarity
        global first_time_main
        global current_page
        print(first_time_main)
        current_page = 0
        query = var_query.get()
        if len(query) != 0:
            record_list.append([True, query])

            query_as_list = [[True, query]]
            ranked_similarity = find_similarity(database_TFIDF,database_IDF, query_as_list)



        # show:
        fresh_table()


        # window_main.update()

    def search_learned():
        global record_list
        global text_list
        global ranked_similarity
        global current_page

        current_page = 0
        query = var_query.get()
        if record_list !=[]:
            ranked_similarity = find_similarity(database_TFIDF, database_IDF, record_list)
        # print(record_list)

        fresh_table()

    def clear_learned():
        global record_list
        global text_list
        global ranked_similarity
        record_list = []

    def fresh_table():
        global current_page

        x = tree.get_children()
        for item in x:
            tree.delete(item)
        # print('')
        # for i in range(10):
        #     # text_list 去头取序号需要减一，columns不去头，直接取从1开始的rank
        #     print('index:', ranked_similarity[i][0], 'score:', ranked_similarity[i][1], 'text:',
        #           columns[ranked_similarity[i][0]])
        # print('')
        start = current_page*30
        if start+30<=len(text_list):
            end = start+30
        else:
            end = len(text_list)
        for i in range(start,end):
            tree.insert('', i,
                        values=(str(ranked_similarity[i][0]),
                                str(ranked_similarity[i][1]),
                                database_records[ranked_similarity[i][0]][0],
                                database_records[ranked_similarity[i][0]][1],
                                database_records[ranked_similarity[i][0]][2],
                                database_records[ranked_similarity[i][0]][3],
                                database_records[ranked_similarity[i][0]][4],
                                database_records[ranked_similarity[i][0]][5],
                                database_records[ranked_similarity[i][0]][6],
                                database_records[ranked_similarity[i][0]][7],
                                database_records[ranked_similarity[i][0]][8],
                                database_records[ranked_similarity[i][0]][9],
                                database_records[ranked_similarity[i][0]][10],
                                database_records[ranked_similarity[i][0]][11],
                                ))
        # print('freshed')

    def calculate_columns_max_lens(columns):
        columns_max_lens =[]
        for i in columns[0]:
            columns_max_lens.append(0)
        for column in columns:
            for i, item in enumerate(column):
                lenTxt = len(item)
                lenTxt_utf8 = len(item.encode('utf-8'))
                item_size = int((lenTxt_utf8 - lenTxt) / 2 + lenTxt)
                if item_size>columns_max_lens[i]:
                    columns_max_lens[i] = item_size
        # print(columns_max_lens)
        return columns_max_lens

    def rel(rank):
        global record_list
        global text_list
        global ranked_similarity
        global first_time_main
        if first_time_main==False:
            record_list.append([True,text_list[ranked_similarity[current_page*30+rank][0] - 1]])
        print('clicked rel ',str(rank))

    def irrel(rank):
        global record_list
        global text_list
        global ranked_similarity
        global first_time_main
        global current_page
        if first_time_main==False:
            record_list.append([False,text_list[ranked_similarity[current_page*30+rank][0] - 1]])
        print('clicked irrel ', str(rank))

    def last_page():
        # global page_n
        # global v
        global current_page
        global first_time_main
        if first_time_main==False:
            if current_page > 0:
                current_page -= 1
            fresh_table()
            # v.set('page:\n' + str(current_page))
            # page_n.update()
        print('page'+str(current_page))


    def next_page():
        global current_page
        global first_time_main
        # global v
        # global page_n

        if first_time_main==False:
            if current_page < (len(text_list) / 30 + 1):
                current_page += 1
            fresh_table()
            # v.set('page:\n' + str(current_page))
            # page_n.update()
        print('page'+str(current_page))

    global first_time_main
    global text_list

    '''initialize'''

    database_name = 'lagou_ITjobs'
    csv_file_name = database_name+'.csv'
    database_records = read_file(csv_file_name)

    text_list =[]
    for i in range(1,database_records.__len__()):
        text_list.append(' '.join(database_records[i]))

    filename = database_name+'_TFIDF_storage.pickle'
    # is_first_time = False
    is_first_time = not os.path.exists(filename)
    if is_first_time:
        print('is the first time')
        database_TFIDF, database_IDF = calculate_text_list_TFIDF(text_list)
        print('store calculated TFIDF')
        with open(filename, 'wb') as f:
            pickle.dump(database_TFIDF, f)
            pickle.dump(database_IDF, f)
    with open(filename, 'rb')as f:
        print('load calculated TFIDF')
        database_TFIDF=pickle.load(f)
        database_IDF=pickle.load(f)
    # max_lens = calculate_columns_max_lens(columns)



    # 新建界面
    # new window
    window_main = tk.Tk()
    w = window_main.winfo_screenwidth()
    h = window_main.winfo_screenheight()
    window_main.geometry("%dx%d" % (w, h))

    # 画布放置图片
    # set image
    canvas = tk.Canvas(window_main, height=h, width=w)
    img=Image.open('background.png')
    img = img.resize((w, h),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image=img)
    image = canvas.create_image(0, 0, anchor='nw', image=img)
    canvas.pack(side='top')

    # canvas = tk.Canvas(window_main, height=h, width=w)
    # imagefile = tk.PhotoImage(file='background.png')
    # image = canvas.create_image(0, 0, anchor='nw', image=img)
    # canvas.pack(side='top')

    search_gif = tk.PhotoImage(file="search_picture1.png")
    imgLabel = tk.Label(window_main, image=search_gif).place(x=50,y=30,width=63,height=30)

    var_query = tk.StringVar()
    entry_query = tk.Entry(window_main, show=None, width=500,bd=4, textvariable=var_query)
    entry_query.place(x=150,y=30,width=500, height=30)

    bt_search = tk.Button(window_main, text='search', command=search)
    bt_search.place(x=670, y=30,width=60, height=30)

    bt_search_learned = tk.Button(window_main, text='search learned', command=search_learned)
    bt_search_learned.place(x=670+80, y=30,width=100, height=30)

    bt_clear_learned = tk.Button(window_main, text='clear learned', command=clear_learned)
    bt_clear_learned.place(x=670+200, y=30,width=100, height=30)

    w_bt=40
    h_bt=15
    w_grid=50
    h_grid=20
    x_base=50
    y_base=107

    tk.Label(window_main, text='rank (in this page)',height=1).place(x=w_grid - 30, y=y_base-30)
    #动态循环绑定注意lambda使用
    # Dynamic Loop Binding Note lambda Use
    for i in range(30):
        temp_y=y_base+i*h_grid
        tk.Label(window_main, text=str(i+1),height=1).place(x=w_grid-30, y=temp_y-3)
        bt_rel = tk.Button(window_main, text='is rel', command = lambda arg=i:rel(arg))
        bt_rel.place(x=x_base, y=temp_y, width=w_bt, height=h_bt)
        bt_irrel = tk.Button(window_main, text='not rel', command = lambda arg=i:irrel(arg))
        bt_irrel.place(x=w_grid+x_base, y=temp_y, width=w_bt, height=h_bt)

    last_page_x = x_base-30
    last_page_y = y_base+30*h_grid+30

    # tried to fresh label but failed
    # global page_n
    # v = tk.StringVar()
    # v.set('page:\n' + str(current_page))
    page_n = tk.Label(window_main, text='page').place(x=last_page_x + 40, y=last_page_y)
    bt_last_page = tk.Button(window_main, text='last', command=lambda: last_page())
    bt_last_page.place(x=last_page_x, y=last_page_y, width=40, height=40)
    bt_next_page = tk.Button(window_main, text='next', command=lambda: next_page())
    bt_next_page.place(x=last_page_x+80, y=last_page_y, width=40, height=40)


    #command = lambda arg=cbname:cbClicked(arg)

    tree = ttk.Treeview(window_main)

    columns_name=[
        ['record index',5],
        ['similarity score',6],
        ['positionName',20],
        ['companyId',6],
        ['companyShortName',20],
        ['companySize',8],
        ['industryField',20],
        ['workYear',8],
        ['city',8],
        ['salary',8],
        ['education',8],
        ['jobNature',8],
        ['companyLabelList',20],
        ['positionAdvantage',20]]
    tree["columns"] = (
        'record index',
        'similarity score',
        'positionName',
        'companyId',
        'companyShortName',
        'companySize',
        'industryField',
        'workYear',
        'city',
        'salary',
        'education',
        'jobNature',
        'companyLabelList',
        'positionAdvantage')
    for i in columns_name:
        tree.column(i[0],width = i[1]*10,anchor='center')
    # i=columns_name[2]
    # tree.column('record index', width=i[1], anchor='center')
    # tree.column('similarity score', width=i[1], anchor='center')
    # tree.column('positionName', width=i[1], anchor='center')
    # tree.column('companyId', width=i[1], anchor='center')
    # tree.column('companyShortName', width=i[1], anchor='center')
    # tree.column('companySize', width=i[1], anchor='center')
    # tree.column('industryField', width=i[1], anchor='center')
    # tree.column('workYear', width=i[1], anchor='center')
    # tree.column('city', width=i[1], anchor='center')
    # tree.column('salary', width=i[1], anchor='center')
    # tree.column('education', width=i[1], anchor='center')
    # tree.column('jobNature', width=i[1], anchor='center')
    # tree.column('companyLabelList', width=i[1], anchor='center')
    # tree.column('positionAdvantage', width=i[1], anchor='center')
    for i in columns_name:
        tree.heading(i[0], text=i[0])

    # xbar = ttk.Scrollbar(window_main, orient='horizontal',command=tree.yview)
    # xbar.place(relx=0.028, rely=0.971, relwidth=0.958, relheight=0.024)
    # tree.configure(xscrollcommand=xbar.set)

    tree.place(x=150,y=80,height=700)

    window_main.title('main window')
    first_time_main =False
    tk.mainloop()


# 退出的函数
def usr_sign_quit():
    window.destroy()


# 登录 注册按钮
bt_login = tk.Button(window, text='login', command=usr_log_in)
bt_login.place(x=140, y=230)
bt_logup = tk.Button(window, text='register', command=usr_sign_up)
bt_logup.place(x=200, y=230)
bt_logquit = tk.Button(window, text='exit', command=usr_sign_quit)
bt_logquit.place(x=280, y=230)
# 主循环
window.mainloop()
