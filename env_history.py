from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
        self.args = args
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history  # prompt, action, state, ...
        
        self.add('state', start_state)

    def add(self, label, value) -> None:
        self._history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self._history = []
    
    def get_prompt(self) -> str:
        coin_name = self.args.dataset
        price_window = self.args.price_window
        reflection_window = self.args.reflection_window
        use_tech = self.args.use_tech
        use_txnstat = self.args.use_txnstat
        delim = '\n"""\n'
        # substitute coin name in the price_s
        price_s = f"You are an {coin_name} cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:" + delim
        # price_s = "You are an ETH cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:" + delim
        for i, item in enumerate(self._history[-price_window * 3:]):
            if item['label'] == 'state':
                state = item['value']
                state_log = f'Open price: {state["open"]:.2f}'
                if use_txnstat:
                    txnstat_dict = state['txnstat']
                    for k, v in txnstat_dict.items():
                        state_log += f', {k}: {v}'
                if use_tech:
                    tech_dict = state['technical']
                    for k, v in tech_dict.items():
                        state_log += f', {k}: {v}'
                price_s += state_log + '\n'
        price_s += delim + 'Write one concise paragraph to analyze the recent information and estimate the statistical trend accordingly.'
        
        # Factual News Agent
        state = self._history[-1]['value']
        # substitute coin name in the factual_news_s
        factual_news_s = f"You are an {coin_name} cryptocurrency trading analyst. You are required to analyze only the factual news, not the subjective news such as someone's comments from following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze and summarize the factual news and estimate the market trend accordingly."
        # factual_news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze only the factual news, not the subjective news such as someone's comments from following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze and summarize the factual news and estimate the market trend accordingly."
        # # Factual Reasoning Agent: this agent is responsible to give reasoning over the factual news, and the reasoning is used for the final trading action
        # factual_reasoning_s = f"You are an ETH cryptocurrency trading analyst. You are required to give the reasoning of the trading suggestions over the above factual news summary. This reasoning will be used for the final trading action."
        # # Subjective News Agent
        # substitute coin name in the subjective_news_s
        subjective_news_s = f"You are an {coin_name} cryptocurrency trading analyst. You are required to analyze only the subjective news such as someone's comments from following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze the subjective news and estimate the market trend accordingly."
        # subjective_news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze only the subjective news such as someone's comments from following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze the subjective news and estimate the market trend accordingly."

        # # Subjective Reasoning Agent: this agent is responsible to give reasoning over the subjective news, and the reasoning is used for the final trading action
        # subjective_reasoning_s = f"You are an ETH cryptocurrency trading analyst. You are required to give the reasoning of the trading suggestions over the above subjective news summary. This reasoning will be used for the final trading action."

        # state = self._history[-1]['value']
        # news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze the following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze the news and estimate the market trend accordingly."

        # substitute coin name in the reflection_s
        reflection_s = f'You are an {coin_name} cryptocurrency trading analyst. Your analysis and action history is given in chronological order:' + delim
        # reflection_s = 'You are an ETH cryptocurrency trading analyst. Your analysis and action history is given in chronological order:' + delim
        for i, item in enumerate(self._history[-reflection_window * 3:]):
            if item['label'] == 'trader_response':
                reflection_s += f'REASONING:\n{item["value"]}\n'
            elif item['label'] == 'action':
                reflection_s += f'ACTION:\n{item["value"]}\n'
            elif item['label'] == 'state':
                reflection_s += f'DAILY RETURN:\n{item["value"]["today_roi"]}\n'
        # reflection_s += delim + 'Reflect on your recent performance and instruct your future trades from a high level, e.g., identify what information is currently more important, and what to be next, like aggresive or conversative. Write one concise paragraph to reflect on your recent trading performance with a focus on the effective strategies and information that led to the most successful outcomes, and the ineffective strategies and information that led to loss of profit. Identify key trends and indicators in the current cryptocurrency market that are likely to influence future trades. Also assess whether a more aggressive or conservative trading approach is warranted.'
        # reflect on the reasoning of the factual and subjective news
        reflection_s += delim + (
        "Reflect on your recent trading performance and provide guidance for future trades, with a specific focus on the balance between factual and subjective information. "
        "Analyze which type of information (factual or subjective) has been more reliable and led to better outcomes. "
        "Identify the most crucial factual data points and subjective insights that have influenced successful trades. "
        "Conversely, pinpoint any instances where over-reliance on either factual or subjective information may have led to suboptimal decisions. "
        "Based on this analysis, recommend whether to weight factual or subjective information more heavily in future trades, providing a clear rationale for your recommendation. "
        "Consider current market trends and indicators, and how they might influence the relative importance of factual versus subjective information. "
        "Conclude with a specific suggestion on the optimal balance between factual and subjective approaches for maximizing trading performance in the current market conditions."
        )

        # substitute coin name in the base_prompt
        base_prompt = f'You are an experienced {coin_name} cryptocurrency trader and you are trying to maximize your overall profit by trading {coin_name}. In each day, you must make an action to buy or sell {coin_name}. You are assisted by a few analysts below and need to decide the final action.'
        # base_prompt = 'You are an experienced ETH cryptocurrency trader and you are trying to maximize your overall profit by trading ETH. In each day, you must make an action to buy or sell ETH. You are assisted by a few analysts below and need to decide the final action.'
        template_s = f"{base_prompt}\n\nON-CHAIN ANALYST REPORT:{delim}{{}}{delim}\nNEWS ANALYST REPORT:{delim}{{}}{delim}\nREFLECTION ANALYST REPORT:{delim}{{}}{delim}\n"
        template_s += '''
Now, provide your response in the following format:

1. Reasoning: Briefly analyze the given reports, considering both factual and subjective elements. Pay attention to the reflection report's guidance on emphasizing factual or subjective reasoning.

2. Factual vs Subjective Weighting: If there's a conflict between factual and subjective information, explain which you favor and why. Assign weights to factual and subjective information (e.g., 0.7 factual, 0.3 subjective) that sum to 1. Justify your weighting based on the reflection report's recommendation to maximize returns.

3. Risk Management: Describe how you're managing risk in light of the current market conditions and the reflection report's insights.

4. Action: Indicate your trading action as a 1-decimal float in the range of [-1,1]. This should reflect your confidence in the market trend, your strategic decision to manage risk, and the balance between factual and subjective reasoning.

Ensure your response is concise yet comprehensive, addressing all the above points.
'''

        return price_s, factual_news_s, subjective_news_s, reflection_s, template_s

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n"
    return query



# from typing import List, Dict


# class EnvironmentHistory:
#     def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
#         self.args = args
#         self._cur_query: str = f'{_get_base_query(base_query, memory)}'
#         self._history: List[Dict[str, str]] = history  # prompt, action, state, ...
        
#         self.add('state', start_state)

#     def add(self, label, value) -> None:
#         self._history += [{
#             'label': label,
#             'value': value,
#         }]

#     def reset(self) -> None:
#         self._history = []
    
#     def get_prompt(self) -> str:
#         price_window = self.args.price_window
#         reflection_window = self.args.reflection_window
#         use_tech = self.args.use_tech
#         use_txnstat = self.args.use_txnstat
#         delim = '\n"""\n'

#         price_s = "You are an ETH cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:" + delim
#         for i, item in enumerate(self._history[-price_window * 3:]):
#             if item['label'] == 'state':
#                 state = item['value']
#                 state_log = f'Open price: {state["open"]:.2f}'
#                 if use_txnstat:
#                     txnstat_dict = state['txnstat']
#                     for k, v in txnstat_dict.items():
#                         state_log += f', {k}: {v}'
#                 if use_tech:
#                     tech_dict = state['technical']
#                     for k, v in tech_dict.items():
#                         state_log += f', {k}: {v}'
#                 price_s += state_log + '\n'
#         price_s += delim + 'Write one concise paragraph to analyze the recent information and estimate the market trend accordingly.'
        
#         state = self._history[-1]['value']
#         news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze the following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze the news and estimate the market trend accordingly."

#         reflection_s = 'You are an ETH cryptocurrency trading analyst. Your analysis and action history is given in chronological order:' + delim
#         for i, item in enumerate(self._history[-reflection_window * 3:]):
#             if item['label'] == 'trader_response':
#                 reflection_s += f'REASONING:\n{item["value"]}\n'
#             elif item['label'] == 'action':
#                 reflection_s += f'ACTION:\n{item["value"]}\n'
#             elif item['label'] == 'state':
#                 reflection_s += f'DAILY RETURN:\n{item["value"]["today_roi"]}\n'
#         reflection_s += delim + 'Reflect on your recent performance and instruct your future trades from a high level, e.g., identify what information is currently more important, and what to be next, like aggresive or conversative. Write one concise paragraph to reflect on your recent trading performance with a focus on the effective strategies and information that led to the most successful outcomes, and the ineffective strategies and information that led to loss of profit. Identify key trends and indicators in the current cryptocurrency market that are likely to influence future trades. Also assess whether a more aggressive or conservative trading approach is warranted.'

#         base_prompt = 'You are an experienced ETH cryptocurrency trader and you are trying to maximize your overall profit by trading ETH. In each day, you will make an action to buy or sell ETH. You are assisted by a few analysts below and need to decide the final action.'
#         template_s = f"{base_prompt}\n\nON-CHAIN ANALYST REPORT:{delim}{{}}{delim}\nNEWS ANALYST REPORT:{delim}{{}}{delim}\nREFLECTION ANALYST REPORT:{delim}{{}}{delim}\n"
#         template_s += 'Now, start your response with your brief reasoning over the given reports. Then, based on the synthesized reports, conclude a clear market trend, emphasizing long-term strategies over short-term gains. Finally, indicate your trading action as a 1-decimal float in the range of [-1,1], reflecting your confidence in the market trend and your strategic decision to manage risk appropriately.'

#         return price_s, news_s, reflection_s, template_s

# def _get_base_query(base_query: str, memory: List[str]) -> str:
#     query = base_query

#     # add memory if it exists
#     if len(memory) > 0:
#         query += '\n\nYour memory for the task below:'
#         for i, m in enumerate(memory):
#             query += f'\nTrial {i}:\n{m.strip()}'
#     query += f"\nHere is the task:\n"
#     return query
