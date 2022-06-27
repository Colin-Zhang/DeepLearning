
import hashlib

def get_file_md5(f):
    m = hashlib.md5()
    while True:
        data = f.read(1024)  #将文件分块读取
        if not data:
            break
        m.update(data)
    return m.hexdigest()


with open('计算机组成原理.docx', 'rb') as f1, open('张晋-年终总结.doc', 'rb') as f2:
    file1_md5 = get_file_md5(f1)
    file2_md5 = get_file_md5(f2)
    print('file1_md5:',file1_md5)
    print('file2_md5:',file2_md5)
    if file1_md5 != file2_md5:
        print('file has changed')
    else:
        print('file not changed')