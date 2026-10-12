from residual_curvature.cli import main

if __name__ == '__main__':
    import sys
    main(['summarize', *sys.argv[1:]])
