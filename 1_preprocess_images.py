import argparse

def tester(param):
    print(param)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default='oui', type=str)
    args = parser.parse_args()
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    tester(args.test)

