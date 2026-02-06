import os, sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import csv
import gc
import re
import time
import shutil
from joblib import Parallel, delayed

BASE_PATH = "/kaggle/input/cert-r4-2/archive" 
LDAP_PATH = os.path.join(BASE_PATH, "LDAP")
ANSWERS_PATH = "/kaggle/input/cert-r4-2-answer/answers" # Hoặc đường dẫn chứa file insiders.csv

def time_convert(inp, mode, real_sd='2010-01-02', sd_monday="2009-12-28"):
    r42_fmt = '%m/%d/%Y %H:%M:%S'
    if mode == 't2dt':
        return datetime.strptime(inp, r42_fmt)
    elif mode == 't2wn': 
        startdate = datetime.strptime(real_sd, '%Y-%m-%d')
        # Tính số ngày chênh lệch và chia 7 để ra số tuần
        return (datetime.strptime(inp, r42_fmt) - startdate).days // 7
    elif mode == 'dt2dn': 
        startdate = datetime.strptime(sd_monday, '%Y-%m-%d')
        return (inp - startdate).days
    elif mode == 'dt2date':
        return inp.strftime("%Y-%m-%d")
    elif mode == 't2date':
        return datetime.strptime(inp, r42_fmt).strftime("%Y-%m-%d")
    return None

WORK_START = datetime.strptime("7:30", "%H:%M").time()
WORK_END = datetime.strptime("17:30", "%H:%M").time()
def is_after_whour(dt): 
    dt_time = dt.time()
    return dt_time < WORK_START or dt_time > WORK_END

def is_weekend(dt):
    return dt.strftime("%w") in ['0', '6']

# --- DATA PREPROCESSING ---
def combine_by_timerange_pandas(dname = 'r4.2', chunk_size=300000):
    all_columns = ['id', 'date', 'user', 'pc', 'type', 
                   'activity', 'url', 'filename', 'content', 
                   'to', 'cc', 'bcc', 'from', 'size', '#att']

    pa_schema = pa.schema([
        ('id', pa.string()),
        ('date', pa.timestamp('ns')), # Timestamp nanosecond (mặc định của pandas)
        ('user', pa.string()),
        ('pc', pa.string()),
        ('type', pa.string()),
        ('activity', pa.string()),
        ('url', pa.string()),
        ('filename', pa.string()),
        ('content', pa.string()),
        ('to', pa.string()),
        ('cc', pa.string()),
        ('bcc', pa.string()),
        ('from', pa.string()),
        ('size', pa.string()),
        ('#att', pa.string())
    ])
    
    allacts = ['device','email','file', 'http','logon']
    http_path = os.path.join(BASE_PATH, 'http.csv')
    with open(http_path, 'r') as f:
        next(f) # Bỏ qua header
        firstline = f.readline().strip()
    
    firstdate_dt = time_convert(firstline.split(',')[1],'t2dt')
    firstdate_dt = firstdate_dt - timedelta(int(firstdate_dt.strftime("%w")))
    firstdate = time_convert(firstdate_dt, 'dt2date')
    
    act_handles = {act: open(os.path.join(BASE_PATH, act+'.csv'), 'r') for act in allacts}
    for h in act_handles.values(): next(h, None) # skip header
    
    lines = {act: act_handles[act].readline() for act in allacts}
    stop = {act: 0 for act in allacts}
    week_index = 0

    print(f"Start processing from date: {firstdate}")
    
    while sum(stop.values()) < 5:
        thisweek_list = []
        week_file_name = f"DataByWeek/{week_index}.parquet"
        writer = None
        for act in allacts:
            while lines[act]:
                tmp = next(csv.reader([lines[act]]))
                if time_convert(tmp[1], 't2wn', real_sd=firstdate) == week_index:
                    # Map columns based on r4.2 format
                    if act == 'email': cols = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', '#att', 'content']
                    elif act in ['logon', 'device']: cols = ['id', 'date', 'user', 'pc', 'activity']
                    elif act in ['http']: cols = ['id', 'date', 'user', 'pc', 'url', 'content']
                    elif act in ['file']: cols = ['id', 'date', 'user', 'pc', 'filename', 'content']
                    
                    entry = dict(zip(cols, tmp))
                    entry['type'] = act
                    thisweek_list.append(entry)
                    lines[act] = act_handles[act].readline()

                    if len(thisweek_list) >= chunk_size:
                        df_chunk = pd.DataFrame(thisweek_list)
                        
                        # Chuẩn hóa cột (Schema Enforcement)
                        for c in all_columns:
                            if c not in df_chunk.columns:
                                df_chunk[c] = None
                        df_chunk = df_chunk[all_columns] # Sắp xếp đúng thứ tự
                        
                        # Xử lý datetime
                        df_chunk['date'] = pd.to_datetime(df_chunk['date'], format="%m/%d/%Y %H:%M:%S")
                        
                        # Chuyển sang PyArrow Table
                        try:
                            table = pa.Table.from_pandas(df_chunk, schema=pa_schema)
                        except Exception as e:
                            print(f"Schema Error at week {week_index}: {e}")
                            # Fallback nếu schema lỗi (hiếm gặp)
                            table = pa.Table.from_pandas(df_chunk)
                        
                        # Khởi tạo writer nếu chưa có (dùng schema của chunk đầu tiên)
                        if writer is None:
                            writer = pq.ParquetWriter(week_file_name, table.schema, compression='snappy')
                        
                        # Ghi chunk và xóa RAM
                        try: writer.write_table(table)
                        except Exception as e:
                            print(f"Error writing chunk at week {week_index}: {e}")

                        del df_chunk, table
                        thisweek_list = [] # Reset buffer
                        gc.collect() # Ép giải phóng RAM
                        
                else: break
            if not lines[act]: stop[act] = 1
        
        if thisweek_list:
            df_chunk = pd.DataFrame(thisweek_list)
            for c in all_columns:
                if c not in df_chunk.columns:
                    df_chunk[c] = None
            df_chunk = df_chunk[all_columns]
            
            df_chunk['date'] = pd.to_datetime(df_chunk['date'], format="%m/%d/%Y %H:%M:%S")
            
            try: table = pa.Table.from_pandas(df_chunk, schema=pa_schema)
            except:
                table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(week_file_name, table.schema, compression='snappy')
            
            writer.write_table(table)
            del df_chunk, table
            gc.collect()

        # Đóng writer để hoàn tất file tuần
        if writer: writer.close()
            
        print(f"Week {week_index} processed.")
        week_index += 1

def process_user_pc(upd, roles): 
    # Xác định PC nào thuộc về người dùng nào
    upd['sharedpc'] = None
    upd['npc'] = upd['pcs'].apply(lambda x: len(x))
    
    # Trường hợp user chỉ dùng 1 PC duy nhất
    upd.at[upd['npc']==1,'pc'] = upd[upd['npc']==1]['pcs'].apply(lambda x: x[0])
    
    # Xử lý các máy tính dùng chung (multi-user)
    multiuser_pcs = np.concatenate(upd[upd['npc']>1]['pcs'].values).tolist()
    set_multiuser_pc = list(set(multiuser_pcs))
    count = {}
    for pc in set_multiuser_pc:
        count[pc] = multiuser_pcs.count(pc)
        
    for u in upd[upd['npc']>1].index:
        sharedpc = upd.loc[u]['pcs']
        count_u_pc = [count[pc] for pc in upd.loc[u]['pcs']]
        # Chọn PC có ít người dùng nhất làm máy chính
        the_pc = count_u_pc.index(min(count_u_pc))
        upd.at[u,'pc'] = sharedpc[the_pc]
        
        # Nếu không phải ITAdmin, các máy còn lại trong danh sách được coi là sharedpc
        if roles.loc[u] != 'ITAdmin':
            sharedpc.remove(sharedpc[the_pc])
            upd.at[u,'sharedpc']= sharedpc
    return upd

def getuserlist(dname = 'r4.2', psycho = True):
    # Đọc dữ liệu nhân sự từ các file LDAP
    allfiles = [os.path.join(LDAP_PATH, f1) for f1 in os.listdir(LDAP_PATH) if os.path.isfile(os.path.join(LDAP_PATH, f1))]
    alluser = {}
    alreadyFired = []
    
    for file in allfiles:
        filename_only = os.path.basename(file) # Lấy tên file (ví dụ: 2010-01.csv)
        af = (pd.read_csv(file, delimiter=',')).values
        employeesThisMonth = []    
        for i in range(len(af)):
            employeesThisMonth.append(af[i][1])
            if af[i][1] not in alluser:
                # Dùng filename_only để split lấy ngày tháng
                alluser[af[i][1]] = af[i][0:1].tolist() + af[i][2:].tolist() + [filename_only.split('.')[0] , np.nan]
        firedEmployees = list(set(alluser.keys()) - set(alreadyFired) - set(employeesThisMonth))
        alreadyFired = alreadyFired + firedEmployees
        for e in firedEmployees:
            alluser[e][-1] = filename_only.split('.')[0]
    
    # Thêm dữ liệu tâm lý học (O-C-E-A-N)
    psycho_path = os.path.join(BASE_PATH, "psychometric.csv")
    if psycho and os.path.isfile(psycho_path):
        p_score = pd.read_csv(psycho_path, delimiter = ',').values
        for id in range(len(p_score)):
            alluser[p_score[id,1]] = alluser[p_score[id,1]] + list(p_score[id,2:])
        df = pd.DataFrame.from_dict(alluser, orient='index')
        df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
    else:
        df = pd.DataFrame.from_dict(alluser, orient='index')
        df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']

    # Chuyển đổi tên người quản lý (supervisor) thành index
    for i in df.index:
        if type(df.loc[i]['sup']) == str:
            sup_matches = df[df['uname'] == df.loc[i]['sup']].index
            sup = sup_matches[0] if len(sup_matches) > 0 else None
        else:
            sup = None
        df.at[i, 'sup'] = sup
        
    # Xác định PC dựa trên log 2 tuần đầu tiên
    w1 = pd.read_parquet("DataByWeek/0.parquet")
    w2 = pd.read_parquet("DataByWeek/1.parquet")
    user_pc_dict = pd.DataFrame(index=df.index)
    user_pc_dict['pcs'] = None  
  
    for u in df.index:
        pc = list(set(w1[w1['user']==u]['pc']) & set(w2[w2['user']==u]['pc']))
        user_pc_dict.at[u,'pcs'] = pc
        
    upd = process_user_pc(user_pc_dict, df['role'])
    df['pc'] = upd['pc']
    df['sharedpc'] = upd['sharedpc']
    return df

def get_mal_userdata(data = 'r4.2', usersdf = None):
    # Lọc danh sách kẻ nội gián cho tập r4.2
    insider_path = os.path.join(ANSWERS_PATH, "insiders.csv")
    listmaluser = pd.read_csv(insider_path)
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset'] == "4.2"]
    
    # Chuyển đổi thời gian bắt đầu/kết thúc sang định dạng datetime
    listmaluser['start'] = pd.to_datetime(listmaluser['start'], format="%m/%d/%Y %H:%M:%S")
    listmaluser['end'] = pd.to_datetime(listmaluser['end'], format="%m/%d/%Y %H:%M:%S")
    
    if usersdf is None:
        usersdf = getuserlist(data)
        
    usersdf['malscene'] = 0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['malacts'] = None
    
    for i in listmaluser.index:
        u_id = listmaluser['user'][i]
        usersdf.loc[u_id, 'mstart'] = listmaluser['start'][i]
        usersdf.loc[u_id, 'mend'] = listmaluser['end'][i]
        usersdf.loc[u_id, 'malscene'] = listmaluser['scenario'][i]
        
        # Đọc chi tiết các hành động độc hại từ folder đáp án r4.2
        scenario_num = str(listmaluser['scenario'][i]) 
        folder_name = f"r4.2-{scenario_num}" 
        file_name = listmaluser['details'][i]
        mal_file_path = os.path.join(ANSWERS_PATH, folder_name, file_name)
        try:
            with open(mal_file_path, 'r') as f:
                malacts = f.read().strip().split("\n")

            if malacts and 'id' in malacts[0].lower(): malacts = malacts[1:]
            malacts = [x.split(',') for x in malacts]
            mal_users = np.array([x[3].strip('"') for x in malacts])
            mal_act_ids = np.array([x[1].strip('"') for x in malacts])
        
            usersdf.at[u_id, 'malacts'] = mal_act_ids[mal_users == u_id]
        
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file tại {mal_file_path}")
            usersdf.at[u_id, 'malacts'] = []
                    
    return usersdf

# --- FEATURE EXTRACTION (Focus: r4.2) ---
def email_process(act, data = 'r4.2'):
    # Phân tách danh sách người nhận từ các trường to, cc, bcc
    receivers = act['to'].split(';')
    if isinstance(act['cc'], str):
        receivers = receivers + act['cc'].split(";")
    
    if isinstance(act['bcc'], str):
        bccreceivers = act['bcc'].split(";")   
    else:
        bccreceivers = []

    exemail = False # Biến đánh dấu nếu có gửi ra ngoài hệ thống dtaa.com
    n_exdes = 0     # Đếm số lượng địa chỉ bên ngoài
    
    # Duyệt qua tất cả người nhận để kiểm tra email ngoài hệ thống
    for i in receivers + bccreceivers:
        if 'dtaa.com' not in i:
            exemail = True
            n_exdes += 1

    n_des = len(receivers) + len(bccreceivers) # Tổng số người nhận
    Xemail = 1 if exemail else 0               # Feature: Có gửi ra ngoài hay không
    n_bccdes = len(bccreceivers)               # Số lượng người nhận trong BCC
    
    # Kiểm tra xem có email ngoài trong danh sách BCC không
    exbccmail = 0
    for i in bccreceivers:
        if 'dtaa.com' not in i:
            exbccmail = 1
            break

    # Tính toán các thông số về văn bản trong nội dung email
    email_text_len = len(act['content'])
    email_text_nwords = act['content'].count(' ') + 1

    # Trả về các đặc trưng (features) dành riêng cho r4.2
    return [
        n_des,              # Tổng số người nhận
        int(act['#att']),   # Số lượng tệp đính kèm (r4.2 sử dụng cột #att)
        Xemail,             # Flag email gửi ra ngoài
        n_exdes,            # Số lượng đích đến bên ngoài
        n_bccdes,           # Số lượng đích đến trong BCC
        exbccmail,          # Flag BCC có chứa địa chỉ ngoài
        int(act['size']),   # Kích thước email
        email_text_len,     # Độ dài ký tự nội dung
        email_text_nwords   # Số lượng từ trong nội dung
    ]

def http_process(act, data = 'r4.2'): 
    # Đặc trưng cơ bản:
    url_len = len(act['url']) # Độ dài URL
    url_depth = max(0, act['url'].count('/') - 2) # Độ sâu của đường dẫn URL
    content_len = len(act['content']) # Độ dài nội dung trang web
    content_nwords = act['content'].count(' ') + 1 # Số lượng từ trong nội dung
    
    # Xử lý lấy tên miền (domain name) từ URL
    domainname = re.findall("//(.*?)/", act['url'])[0]
    domainname = domainname.replace("www.", "")
    dn = domainname.split(".")
    
    # Rút gọn domain nếu là subdomain (loại trừ các domain phổ biến)
    if len(dn) > 2 and not any([x in domainname for x in ["google.com", '.co.uk', '.co.nz', 'live.com']]):
        domainname = ".".join(dn[-2:])

    # Phân loại website (Category r): 
    # other: 1, socnet: 2, cloud: 3, job: 4, leak: 5, hack: 6
    
    # Nhóm 3: Cloud / Lưu trữ trực tuyến
    if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
        r = 3
    # Nhóm 5: Các trang web rò rỉ dữ liệu
    elif domainname in ['wikileaks.org', 'freedom.press', 'theintercept.com']:
        r = 5
    # Nhóm 2: Mạng xã hội
    elif domainname in ['facebook.com', 'twitter.com', 'plus.google.com', 'instagr.am', 'instagram.com',
                        'flickr.com', 'linkedin.com', 'reddit.com', 'about.com', 'youtube.com', 'pinterest.com',
                        'tumblr.com', 'quora.com', 'vine.co', 'match.com', 't.co']:
        r = 2
    # Nhóm 4: Tìm kiếm việc làm
    elif domainname in ['indeed.com', 'monster.com', 'careerbuilder.com', 'simplyhired.com']:
        r = 4
    elif ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) \
    or ('aol.com' in domainname and ("recruit" in act['url'] or "job" in act['url'])):
        r = 4
    # Nhóm 6: Công cụ tấn công/giám sát/keylogger
    elif (domainname in ['webwatchernow.com', 'actionalert.com', 'relytec.com', 'refog.com', 'wellresearchedreviews.com',
                         'softactivity.com', 'spectorsoft.com', 'best-spy-soft.com']):
        r = 6
    elif ('keylog' in domainname):
        r = 6
    # Nhóm 1: Các trang web khác
    else:
        r = 1
    # Trả về 5 đặc trưng dành riêng cho r4.2
    return [r, url_len, url_depth, content_len, content_nwords]

def file_process(act, data = 'r4.2'):
    # Lấy phần mở rộng của file (ví dụ: .doc, .txt)
    if "." in act['filename']:
        ftype = act['filename'].split(".")[-1].lower()
    else:
        ftype = "unknown"

    # Xác định loại ổ đĩa: C (nội bộ): 1, R (ổ đĩa mạng/rời): 2, Khác: 0
    disk = 1 if act['filename'][0] == 'C' else 0
    if act['filename'][0] == 'R': 
        disk = 2
    
    # Độ sâu của thư mục (số lượng dấu gạch chéo ngược)
    file_depth = act['filename'].count('\\')

    # Đặc trưng từ nội dung file
    fsize = len(act['content']) # Kích thước dựa trên độ dài nội dung
    f_nwords = act['content'].count(' ') + 1 # Số lượng từ trong nội dung

    # Phân loại nhóm định dạng file (r):
    # nén: 2, ảnh: 3, văn bản/pdf: 4, text/config: 5, thực thi: 6, khác: 1
    if ftype in ['zip', 'rar', '7z']:
        r = 2
    elif ftype in ['jpg', 'png', 'bmp']:
        r = 3
    elif ftype in ['doc', 'docx', 'pdf']:
        r = 4
    elif ftype in ['txt', 'cfg', 'rtf']:
        r = 5
    elif ftype in ['exe', 'sh']:
        r = 6
    else:
        r = 1
    # Trả về 5 đặc trưng tiêu chuẩn của r4.2 cho hành động file
    return [r, fsize, f_nwords, disk, file_depth]

def from_pc(act, ul):
    #code: 0,1,2,3:  own pc, sharedpc, other's pc, supervisor's pc
    user_pc = ul.loc[act['user']]['pc']
    act_pc = act['pc']
    if act_pc == user_pc:
        return (0, act_pc) #using normal PC
    elif ul.loc[act['user']]['sharedpc'] is not None and act_pc in ul.loc[act['user']]['sharedpc']:
        return (1, act_pc)
    elif ul.loc[act['user']]['sup'] is not None and act_pc == ul.loc[ul.loc[act['user']]['sup']]['pc']:
        return (3, act_pc)
    else:
        return (2, act_pc)

def process_week_num(week, users, userlist = 'all', data = 'r4.2', chunk_size=300000):
    user_dict = {idx: i for (i, idx) in enumerate(users.index)}        
    
    # Đọc dữ liệu nguồn (Input)
    file_path = f"DataByWeek/{week}.parquet"
    if not os.path.exists(file_path): return        
    acts_week = pd.read_parquet(file_path)
    
    # Sort theo thời gian là bắt buộc cho logic sau này
    acts_week.sort_values('date', ascending = True, inplace = True)
    
    if userlist == 'all': userlist = set(acts_week.user)
    
    # Định nghĩa tên cột cho DataFrame kết quả
    col_names = ['user','day','act','pc','time', 'usb_dur',
                 'file_type', 'file_len', 'file_nwords', 'disk', 'file_depth',
                 'http_type', 'url_len','url_depth', 'http_c_len', 'http_c_nwords',
                 'n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords',
                 'mal_act','insider']
    
    # Chuẩn bị ghi file Streaming
    save_path = f"NumDataByWeek/{week}_num.parquet"
    writer = None
    
    # Buffer chứa dữ liệu tạm
    buffer_data = [] 
    buffer_meta = [] # Chứa actid, pcid, time_stamp
    buffer_count = 0
    
    for u in userlist:
        df_acts_u = acts_week[acts_week.user == u]
        if len(df_acts_u) == 0: continue

        # --- LOGIC TÍNH TOÁN GIỮ NGUYÊN ---
        mal_u = 0         
        if users.loc[u].malscene > 0:
            start_week, end_week = min(acts_week.date), max(acts_week.date) # Lấy range tuần để check
            if start_week <= users.loc[u].mend and users.loc[u].mstart <= end_week:
                mal_u = users.loc[u].malscene
        
        list_uacts = df_acts_u.type.tolist()       
        list_activity = df_acts_u.activity.tolist()
        
        list_uacts = [list_activity[i].strip().lower() if (isinstance(list_activity[i], str) and \
                      list_activity[i].strip() in ['Logon', 'Logoff', 'Connect', 'Disconnect']) \
                      else list_uacts[i] for i in range(len(list_uacts))]  
        
        uacts_mapping = {'logon':1, 'logoff':2, 'connect':3, 'disconnect':4, 'http':5, 'email':6, 'file':7}
        list_uacts_num = [uacts_mapping.get(x, 0) for x in list_uacts]

        # Xử lý từng dòng hành động của User này
        for i in range(len(df_acts_u)):
            pc, _ = from_pc(df_acts_u.iloc[i], users)
            dt = df_acts_u.iloc[i]['date']
            if is_weekend(dt): act_time = 4 if is_after_whour(dt) else 3
            else: act_time = 2 if is_after_whour(dt) else 1
            
            device_f = [0]      
            file_f = [0]*5      
            http_f = [0]*5      
            email_f = [0]*9     
            
            curr_act = list_uacts[i]
            if curr_act == 'file': file_f = file_process(df_acts_u.iloc[i], data=data)
            elif curr_act == 'email': email_f = email_process(df_acts_u.iloc[i], data=data)
            elif curr_act == 'http': http_f = http_process(df_acts_u.iloc[i], data=data)
            elif curr_act == 'connect':
                tmp_future = df_acts_u.iloc[i:]
                disconnect_acts = tmp_future[(tmp_future['activity'] == 'Disconnect\n') & \
                                             (tmp_future['pc'] == df_acts_u.iloc[i]['pc'])]
                if len(disconnect_acts) > 0:
                    distime = disconnect_acts.iloc[0]['date']
                    tmp_td = distime - dt
                    connect_dur = tmp_td.days*24*3600 + tmp_td.seconds
                else: connect_dur = -1
                device_f = [connect_dur]
                
            is_mal_act = 0
            if mal_u > 0 and df_acts_u.iloc[i]['id'] in users.loc[u]['malacts']: 
                is_mal_act = 1

            # Tạo row dữ liệu
            row_vals = [user_dict[u], time_convert(dt, 'dt2dn'), list_uacts_num[i], pc, act_time] \
                             + device_f + file_f + http_f + email_f + [is_mal_act, mal_u]
            
            buffer_data.append(row_vals)
            buffer_meta.append([df_acts_u.index[i], df_acts_u.iloc[i]['pc'], dt])
            buffer_count += 1
            
            # --- KIỂM TRA CHUNK SIZE ĐỂ GHI ---
            if buffer_count >= chunk_size:
                # Tạo DataFrame từ buffer
                df_chunk = pd.DataFrame(buffer_meta, columns=['actid','pcid','time_stamp'])
                df_data = pd.DataFrame(buffer_data, columns=col_names)
                
                # Gộp 2 phần meta và data
                df_final = pd.concat([df_chunk, df_data], axis=1)
                
                # Ép kiểu int cho các cột data để tiết kiệm bộ nhớ
                df_final[col_names] = df_final[col_names].astype(int)
                
                # Convert sang PyArrow Table
                table = pa.Table.from_pandas(df_final)
                
                if writer is None:
                    writer = pq.ParquetWriter(save_path, table.schema, compression='snappy')
                
                writer.write_table(table)
                
                # Dọn dẹp RAM
                del df_chunk, df_data, df_final, table
                buffer_data = []
                buffer_meta = []
                buffer_count = 0
                import gc; gc.collect()

    # --- GHI NỐT PHẦN CÒN DƯ ---
    if buffer_count > 0:
        df_chunk = pd.DataFrame(buffer_meta, columns=['actid','pcid','time_stamp'])
        df_data = pd.DataFrame(buffer_data, columns=col_names)
        df_final = pd.concat([df_chunk, df_data], axis=1)
        df_final[col_names] = df_final[col_names].astype(int)
        
        table = pa.Table.from_pandas(df_final)
        
        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema, compression='snappy')
        
        writer.write_table(table)
        del df_final, table
        import gc; gc.collect()

    if writer:
        writer.close()
    # Xóa biến lớn input
    del acts_week
    gc.collect()
    
# --- SESSION LOGIC ---
def get_sessions(uw, first_sid=0):
    sessions = {}
    open_sessions = {}
    sid = 0
    
    for i in uw.index:
        current_pc = uw.loc[i]['pcid']
        act_type = uw.loc[i]['act'] # 1: Logon, 2: Logoff
        
        if current_pc in open_sessions:
            if act_type == 2: # Logoff
                open_sessions[current_pc][2] = 1 # end_with logoff
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
            elif act_type == 1: # New Logon on same PC without logoff
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
                open_sessions[current_pc] = [current_pc, 1, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
            else:
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
        else:
            start_status = 1 if act_type == 1 else 2
            open_sessions[current_pc] = [current_pc, start_status, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]

    for pc in open_sessions:
        sessions[sid] = [first_sid + sid] + open_sessions[pc]
        sid += 1

    return sessions

def get_u_features_dicts(ul, data = 'r4.2'):
    """Tạo từ điển ánh xạ thông tin người dùng cho r4.2"""
    ufdict = {}
    # r4.2 không có cột 'project'
    list_uf = ['role', 'b_unit', 'f_unit', 'dept', 'team']
    
    for f in list_uf:
        ul[f] = ul[f].astype(str)
        tmp = list(set(ul[f]))
        tmp.sort()
        # Ánh xạ mỗi giá trị chữ sang một số nguyên
        ufdict[f] = {idx: i for i, idx in enumerate(tmp)}
    return (ul, ufdict, list_uf)

def proc_u_features(uf, ufdict, list_f = None, data = 'r4.2'):
    """Chuyển đổi đặc trưng của một user cụ thể sang dạng số"""
    if type(list_f) != list:
        # Mặc định danh sách cột cho r4.2
        list_f = ['role', 'b_unit', 'f_unit', 'dept', 'team']

    out = []
    for f in list_f:
        # Lấy giá trị số từ từ điển ánh xạ đã tạo
        out.append(ufdict[f][uf[f]])
    return out

def f_stats_calc(ud, fn, stats_f, countonly_f = {}, get_stats = False):
    """Tính toán các chỉ số thống kê cơ bản (Mean, Min, Max,...)"""
    f_count = len(ud)
    r = []
    f_names = []
    
    # Tính thống kê cho các trường dữ liệu số (ví dụ: độ dài nội dung, kích thước file)
    for f in stats_f:
        inp = ud[f].values
        if get_stats:
            if f_count > 0:
                r += [np.min(inp), np.max(inp), np.median(inp), np.mean(inp), np.std(inp)]
            else: 
                r += [0.0, 0.0, 0.0, 0.0, 0.0]
            f_names += [fn+'_min_'+f, fn+'_max_'+f, fn+'_med_'+f, fn+'_mean_'+f, fn+'_std_'+f]
        else:
            # Mặc định chỉ lấy trung bình nếu get_stats = False
            r += [np.mean(inp)] if f_count > 0 else [0.0]
            f_names += [fn+'_mean_'+f]
        
    # Đếm số lượng xuất hiện của các giá trị cụ thể (ví dụ: số lần dùng PC lạ)
    for f in countonly_f:
        for v in countonly_f[f]:
            r += [sum(ud[f].values == v)]
            f_names += [fn+'_n-'+f+str(v)]
            
    return (f_count, r, f_names)

def f_calc_subfeatures(ud, fname, filter_col, filter_vals, filter_names, sub_features, countonly_subfeatures):
    """Tính toán đặc trưng cho các nhóm con (ví dụ: chỉ tính riêng cho email gửi ra ngoài)"""
    # 1. Tính tổng quát cho toàn bộ hành động
    [n, stats, fnames] = f_stats_calc(ud, fname, sub_features, countonly_subfeatures)
    allf = [n] + stats
    allf_names = ['n_' + fname] + fnames
    
    # 2. Lọc và tính riêng cho từng phân loại con (filter)
    for i in range(len(filter_vals)):
        filtered_data = ud[ud[filter_col] == filter_vals[i]]
        [n_sf, sf_stats, sf_fnames] = f_stats_calc(filtered_data, filter_names[i], sub_features, countonly_subfeatures)
        allf += [n_sf] + sf_stats
        allf_names += [fname + '_n_' + filter_names[i]] + [fname + '_' + x for x in sf_fnames]
        
    return (allf, allf_names)

def f_calc(ud, mode = 'session', data = 'r4.2'):
    # Khởi tạo các biến cơ bản
    n_weekendact = (ud['time'] == 3).sum()
    is_weekend = 1 if n_weekendact > 0 else 0
    
    # Đối với mode 'session', không đếm số lượng theo PC vì mỗi session thường gắn với 1 PC
    all_countonlyf = {}
    
    # 1. Thống kê chung cho tất cả hành động trong session
    [all_f, all_f_names] = f_calc_subfeatures(ud, 'allact', None, [], [], [], all_countonlyf)
    
    # 2. Thống kê hành động Logon (act == 1)
    logon_countonlyf = {}
    logon_statf = []
    [all_logonf, all_logonf_names] = f_calc_subfeatures(ud[ud['act']==1], 'logon', None, [], [], logon_statf, logon_countonlyf)
    
    # 3. Thống kê thiết bị USB (act == 3) - r4.2 chỉ có usb_dur
    device_countonlyf = {}
    device_statf = ['usb_dur']
    [all_devicef, all_devicef_names] = f_calc_subfeatures(ud[ud['act']==3], 'usb', None, [], [], device_statf, device_countonlyf)
    
    # 4. Thống kê File (act == 7) - r4.2 lược bỏ to_usb, from_usb, file_act
    # r4.2: disk 0: unknown, 1: C, 2: R
    file_countonlyf = {'disk':[0, 1, 2]}
    (all_filef, all_filef_names) = f_calc_subfeatures(
        ud[ud['act']==7], 'file', 'file_type', [1,2,3,4,5,6], 
        ['otherf','compf','phof','docf','txtf','exef'], 
        ['file_len', 'file_depth', 'file_nwords'], 
        file_countonlyf
    )
    
    # 5. Thống kê Email (act == 6)
    email_stats_f = ['n_des', 'n_atts', 'n_exdes', 'n_bccdes', 'email_size', 'email_text_slen', 'email_text_nwords']
    mail_countonlyf = {'Xemail':[1], 'exbccmail':[1]}
    # r4.2 không có bộ lọc send/receive trong file email.csv thô
    (all_emailf, all_emailf_names) = f_calc_subfeatures(ud[ud['act']==6], 'email', None, [], [], email_stats_f, mail_countonlyf)
    
    # 6. Thống kê HTTP (act == 5)
    http_count_subf = {} # Lược bỏ pc và http_act (chỉ có ở r6)
    (all_httpf, all_httpf_names) = f_calc_subfeatures(
        ud[ud['act']==5], 'http', 'http_type', [1,2,3,4,5,6], 
        ['otherf','socnetf','cloudf','jobf','leakf','hackf'], 
        ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], 
        http_count_subf
    )
        
    # Xác định thông tin Insider (mal_act)
    numActs = all_f[0]
    mal_u = 0
    if (ud['mal_act']).sum() > 0:
        tmp = list(set(ud['insider']))
        if len(tmp) > 1 and 0.0 in tmp:
            tmp.remove(0.0)
        mal_u = tmp[0]
        
    # Gộp tất cả vector đặc trưng của session
    features_tmp = all_f + all_logonf + all_devicef + all_filef + all_emailf + all_httpf
    fnames_tmp = all_f_names + all_logonf_names + all_devicef_names + all_filef_names + all_emailf_names + all_httpf_names
    
    return [numActs, is_weekend, features_tmp, fnames_tmp, mal_u]

def session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf):
    # Lấy thông tin ngày thực hiện hành động đầu tiên trong session
    d = ud.iloc[0]['day']
    
    # Tính toán tỷ lệ thời gian hoạt động trong session
    # 1: Giờ hành chính, 2: Ngoài giờ, 3: Cuối tuần, 4: Đêm cuối tuần
    total_acts = len(ud)
    perworkhour = sum(ud['time'] == 1) / total_acts
    perafterhour = sum(ud['time'] == 2) / total_acts
    perweekend = sum(ud['time'] == 3) / total_acts
    perweekendafterhour = sum(ud['time'] == 4) / total_acts
    
    # Tính toán thời lượng và mốc thời gian của session
    st_timestamp = min(ud['time_stamp'])
    end_timestamp = max(ud['time_stamp'])
    
    # Thời lượng session tính bằng phút
    s_dur = (end_timestamp - st_timestamp).total_seconds() / 60
    
    # Giờ bắt đầu và kết thúc (quy đổi sang số thực, ví dụ 7:30 = 7.5)
    s_start = st_timestamp.hour + st_timestamp.minute / 60
    s_end = end_timestamp.hour + end_timestamp.minute / 60
    
    # Timestamp chuẩn (Epoch time)
    starttime = st_timestamp.timestamp()
    endtime = end_timestamp.timestamp()
    
    # Đếm số ngày session kéo dài (thường là 1)
    n_days = len(set(ud['day']))
    
    # Gọi hàm f_calc đã lọc cho r4.2 để lấy các đặc trưng thống kê hoạt động
    tmp = f_calc(ud, mode, data)
    
    # Tạo vector instance hoàn chỉnh cho session
    # Kết hợp: Thông tin thời gian + Đặc trưng Session + Thông tin User + Đặc trưng Hoạt động + Nhãn Insider
    session_instance = [
        starttime, endtime, v, sinfo[0], d, week, ud.iloc[0]['pc'], 
        perworkhour, perafterhour, perweekend, perweekendafterhour, 
        n_days, s_dur, 
        sinfo[6], # n_concurrent_login (số lượng login đồng thời)
        sinfo[2], # start_with (bắt đầu bằng logon hay không)
        sinfo[3], # end_with (kết thúc bằng logoff hay không)
        s_start, s_end
    ] + \
    (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']]).tolist() + \
    tmp[2] + [tmp[4]] # tmp[2] là features, tmp[4] là nhãn insider
    
    return (session_instance, tmp[3]) # Trả về instance và danh sách tên cột (tmp[3])

def to_csv(week, mode, data, ul, uf_dict, list_uf, chunk_size=300000):
    # Khởi tạo từ điển ánh xạ user ID
    user_dict = {i : idx for (i, idx) in enumerate(ul.index)} 
    
    # Thiết lập ID session duy nhất dựa trên tuần
    # Ví dụ: tuần 1 sẽ bắt đầu từ 100000, tuần 2 từ 200000
    first_sid = week * 100000 
    
    # Định nghĩa các cột thông tin cơ bản cho Session r4.2
    cols2a = [
        'starttime', 'endtime', 'user', 'sessionid', 'day', 'week', 'pc', 
        'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 
        'n_days', 'duration', 'n_concurrent_sessions', 'start_with', 'end_with', 
        'ses_start', 'ses_end'
    ] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    
    cols2b = ['insider']        

    # Đọc dữ liệu số đã xử lý của tuần hiện tại
    w = pd.read_parquet(f"NumDataByWeek/{week}_num.parquet")
    usnlist = list(set(w['user'].astype('int').values))
    
    # Tạo bảng thông tin User tĩnh cho tuần này
    cols_u = ['week'] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'] 
    uwdict = {}
    for v in user_dict:
        if v in usnlist:
            is_ITAdmin = 1 if ul.loc[user_dict[v], 'role'] == 'ITAdmin' else 0
            # Mã hóa các thông tin category (role, dept...) thành số
            u_feats = proc_u_features(ul.loc[user_dict[v]], uf_dict, list_uf, data=data)
            # Lấy chỉ số tâm lý OCEAN
            ocean = (ul.loc[user_dict[v], ['O', 'C', 'E', 'A', 'N']]).tolist()
            # Lấy nhãn insider
            insider_label = int(list(set(w[w['user'] == v]['insider']))[0])
            
            uwdict[v] = [week] + u_feats + [is_ITAdmin] + ocean + [insider_label]
            
    uw = pd.DataFrame.from_dict(uwdict, orient='index', columns=cols_u)    
    
    output_file = f"tmp/{week}{mode}.parquet"
    writer = None
    towrite_buffer = []
    full_columns = None
    
    # Duyệt qua từng User
    for v in user_dict:
        if v in usnlist:
            uactw = w[w['user'] == v]
            
            # Get sessions
            sessions = get_sessions(uactw, first_sid)
            first_sid += len(sessions)

            all_sess_info = list(sessions.values())
            # Lưu ý: sinfo[4] là start_time, sinfo[5] là end_time (dạng timestamp hoặc object)
            starts = np.array([s[4] for s in all_sess_info])
            ends = np.array([s[5] for s in all_sess_info])
            
            # Tính concurrency cho từng session
            # Logic: Session A bị coi là concurrent nếu nó trùng thời gian với bất kỳ session B nào khác (PC khác)
            concurrent_counts = []
            for k in range(len(all_sess_info)):
                cur_start = starts[k]
                cur_end = ends[k]
                # Đếm số lượng session có khoảng thời gian giao nhau (Overlap)
                # Điều kiện overlap: (Start_B < End_A) AND (End_B > Start_A)
                overlaps = np.sum((starts < cur_end) & (ends > cur_start))
                # overlaps sẽ luôn >= 1 (trùng với chính nó). 
                concurrent_counts.append(overlaps)

            # Cập nhật lại giá trị vào dict sessions
            for idx, key in enumerate(sessions):
                sessions[key][6] = concurrent_counts[idx] # Ghi đè vào vị trí dummy số 1 cũ
            
            for s in sessions:
                sinfo = sessions[s]
                ud = uactw.loc[sinfo[7]]
                
                if len(ud) > 0:                     
                    # Tính feature
                    session_instance, i_fnames = session_instance_calc(
                        ud, sinfo, week, mode, data, uw, v, list_uf
                    )
                    
                    # Nếu là lần đầu tiên, xác định danh sách cột đầy đủ
                    if writer is None:
                         full_columns = cols2a + i_fnames + cols2b
                    
                    towrite_buffer.append(session_instance)
                    
                    # --- FLUSH BUFFER NẾU ĐẦY ---
                    if len(towrite_buffer) >= chunk_size:
                        df_chunk = pd.DataFrame(towrite_buffer, columns=full_columns)
                        
                        # Convert sang Table và ghi
                        table = pa.Table.from_pandas(df_chunk)
                        
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                        
                        writer.write_table(table)
                        
                        # Dọn dẹp RAM
                        del df_chunk, table
                        towrite_buffer = [] # Reset buffer
                        gc.collect()

    # --- GHI PHẦN CÒN DƯ (Buffer còn lại) ---
    if len(towrite_buffer) > 0:
        try:
            df_chunk = pd.DataFrame(towrite_buffer, columns=full_columns)
            table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            
            writer.write_table(table)
        except UnboundLocalError:
            # Trường hợp không có session nào được tạo ra
            pass

    if writer:
        writer.close()
    
    # Xóa biến lớn để giải phóng RAM cho joblib process khác
    del w, uw
    gc.collect()

if __name__ == "__main__":
    # 1. Kiểm tra thư mục hiện tại có phải là r4.2 không
    dname = 'r4.2'
    # 2. Tạo các thư mục tạm và thư mục chứa kết quả
    for folder in ["tmp", "ExtractedData", "DataByWeek", "NumDataByWeek"]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    # Cấu hình số luồng xử lý song song (mặc định 8)
    numCores = 4
    if len(sys.argv) > 1:
        try:
            # Cố gắng đọc tham số nếu người dùng truyền số vào
            numCores = int(sys.argv[1])
        except ValueError:
            # Nếu gặp lỗi (do sys.argv[1] là '-f' của Jupyter), bỏ qua và dùng mặc định
            print(f"Running in Notebook mode. Using default numCores = {numCores}")
        
    # r4.2 có 73 tuần dữ liệu
    numWeek = 73 
    st = time.time()
    
    #### Bước 1: Phân tách dữ liệu nguồn theo từng tuần
    combine_by_timerange_pandas(dname)
    print(f"Step 1 - Separate data by week - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 2: Lấy danh sách nhân sự và dán nhãn Insider
    users = get_mal_userdata(dname)
    print(f"Step 2 - Get user list & Insider labels - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 3: Chuyển đổi log thô sang dạng số (Numerical)
    Parallel(n_jobs=numCores)(delayed(process_week_num)(i, users, data=dname) for i in range(numWeek))
    print(f"Step 3 - Numerical conversion - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 4: Trích xuất đặc trưng theo SESSION và xuất ra CSV
    mode = 'session'
    (ul, uf_dict, list_uf) = get_u_features_dicts(users, data=dname)
    
    # Chạy song song việc gom nhóm session và tính toán đặc trưng thống kê
    Parallel(n_jobs=numCores)(delayed(to_csv)(i, mode, dname, ul, uf_dict, list_uf) for i in range(numWeek))

    # Gộp tất cả các file pickle tạm thời trong 'tmp/' thành một file CSV duy nhất
    output_file = f'ExtractedData/{mode}_{dname}.parquet'
    print(f"Starting to merge files into {output_file}...")
    writer = None
    ref_schema = None
    for w in range(numWeek):
        week_file = f"tmp/{w}{mode}.parquet"
        if os.path.exists(week_file):
            # Đọc file pickle tuần hiện tại
            df_chunk = pd.read_parquet(week_file)
            
            if writer is None:
                # Đây là chunk đầu tiên (thường là tuần 0 hoặc 1) -> Làm chuẩn
                table = pa.Table.from_pandas(df_chunk)
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                writer.write_table(table)
                # Lưu lại kiểu dữ liệu của pandas để ép các chunk sau
                ref_dtypes = df_chunk.dtypes
            else:
                try:
                    df_chunk = df_chunk.astype(ref_dtypes)
                except Exception as e:
                    print(f"Warning: Could not cast types for week {w}. Reason: {e}")
                
                table = pa.Table.from_pandas(df_chunk)
                try: 
                    writer.write_table(table)
                except Exception as e:
                    print(f"Error writing week {w}: {e}")
        else: pass
    # Đóng writer để hoàn tất file
    if writer:
        writer.close()
    print(f'Step 4 - Extracted {mode} data to {output_file}. Time (mins): {(time.time()-st)/60:.2f}')

    #### Bước 5: Dọn dẹp thư mục tạm
    print("Cleaning up temporary files...")
    for x in ["tmp", "DataByWeek", "NumDataByWeek"]:
        if os.path.exists(x): shutil.rmtree(x)