import argparse

args = argparse.Namespace()
my_dict = {'test': {'a': 1, 'b': 2}, 'key': 'value', 'k': 'v'}

for k, v in my_dict.items():
    print(k , v)
    if isinstance(v, dict):
        setattr(args, k, argparse.Namespace(**v))
    else:
        setattr(args, k, v)

print(args)
print(args.test.a)
print(args.key)