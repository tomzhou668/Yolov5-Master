import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--happy", type=int, default=100)
parse.add_argument("--add", type=str, default="nice")
print(parse.parse_args())