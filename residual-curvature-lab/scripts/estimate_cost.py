from residual_curvature.cli import main

if __name__ == '__main__':
    import sys
    main(['estimate', *sys.argv[1:]])
