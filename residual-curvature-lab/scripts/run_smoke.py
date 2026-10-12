from residual_curvature.cli import main

if __name__ == '__main__':
    import sys
    main(['smoke', *sys.argv[1:]])
