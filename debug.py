from run_agent import get_parser, main

if __name__ == '__main__':
    parser = get_parser()
    debug_args = '--use_tech 1 --use_txnstat 1 --use_news 1 --use_reflection 1 --dataset btc --starting_date 2023-05-21 --ending_date 2023-05-23'.split(' ')
    args = parser.parse_args(debug_args)
    main(args)
