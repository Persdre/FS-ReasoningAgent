# btc
## o1-mini bull
python -u run_agent.py --model o1-mini --dataset btc --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/btc-bull-o1-mini-reasoning.out 2>&1

## o1-mini bear
python -u run_agent.py --model o1-mini --dataset btc --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/btc-bear-o1-mini-reasoning.out 2>&1

## 4o bull
python -u run_agent.py --model gpt-4o --dataset btc --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/btc-bull-4o-reasoning.out 2>&1
## 4o bear
python -u run_agent.py --model gpt-4o --dataset btc --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/btc-bear-4o-reasoning.out 2>&1  

## 3.5 bull
python -u run_agent.py --model gpt-3.5-turbo --dataset btc --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/btc-bull-3.5-reasoning.out 2>&1
## 3.5 bear
python -u run_agent.py --model gpt-3.5-turbo --dataset btc --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/btc-bear-3.5-reasoning.out 2>&1


# eth
## o1-mini bull
python -u run_agent.py --model o1-mini --dataset eth --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/eth-bull-o1-mini-reasoning.out 2>&1
## o1-mini bear
python -u run_agent.py --model o1-mini --dataset eth --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/eth-bear-o1-mini-reasoning.out 2>&1


## 4o bull
python -u run_agent.py --model gpt-4o --dataset eth --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/eth-bull-4o-reasoning.out 2>&1
## 4o bear
python -u run_agent.py --model gpt-4o --dataset eth --starting_date 2024-05-27 --ending_date 2024-07-08 &>updated_reasoning/eth-bear-4o-reasoning.out 2>&1

## 4 bull
python -u run_agent.py --model gpt-4 --dataset eth --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/eth-bull-4-reasoning.out 2>&1
## 4 bear
python -u run_agent.py --model gpt-4 --dataset eth --starting_date 2024-05-27 --ending_date 2024-07-08 &>updated_reasoning/eth-bear-4-reasoning.out 2>&1

## 3.5 bull
python -u run_agent.py --model gpt-3.5-turbo --dataset eth --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/eth-bull-3.5-reasoning.out 2>&1
# ## 3.5 bear
python -u run_agent.py --model gpt-3.5-turbo --dataset eth --starting_date 2024-05-27 --ending_date 2024-07-08 &>updated_reasoning/eth-bear-3.5-reasoning.out 2>&1


# sol
## o1-mini bull
python -u run_agent.py --model o1-mini --dataset sol --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/sol-bull-o1-mini-reasoning.out 2>&1
## o1-mini bear
python -u run_agent.py --model o1-mini --dataset sol --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/sol-bear-o1-mini-reasoning.out 2>&1

## 4o bull
python -u run_agent.py --model gpt-4o --dataset sol --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/sol-bull-4o-reasoning.out 2>&1
## 4o bear
python -u run_agent.py --model gpt-4o --dataset sol --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/sol-bear-4o-reasoning.out 2>&1

## 4 bull
python -u run_agent.py --model gpt-4 --dataset sol --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/sol-bull-4-reasoning.out 2>&1
## 4 bear
python -u run_agent.py --model gpt-4 --dataset sol --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/sol-bear-4-reasoning.out 2>&1

## 3.5 bull
python -u run_agent.py --model gpt-3.5-turbo --dataset sol --starting_date 2024-01-24 --ending_date 2024-03-13 &>updated_reasoning/sol-bull-3.5-reasoning.out 2>&1
## 3.5 bear
python -u run_agent.py --model gpt-3.5-turbo --dataset sol --starting_date 2024-05-21 --ending_date 2024-07-08 &>updated_reasoning/sol-bear-3.5-reasoning.out 2>&1