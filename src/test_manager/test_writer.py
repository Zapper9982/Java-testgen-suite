import os

def write_test(class_name: str, test_code: str, out_dir='src/test/java'):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{class_name}Test.java")
    with open(fname, 'w') as wf:
        wf.write(test_code)
