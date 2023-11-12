
from typing import List, Dict
from numbers import Number
import erniebot
import re
from collections import deque
import traceback

class AIChat:
    def __init__(
                self,
                messages:List[Dict[str, str]] = [], # 无限储存msg，群消息保存在group_context里
                conversation_id: str = "", # 给临时会话用的id
                qq: str = "", # 临时会话使用者的qq号，无效参数，仅用来判断是否临时会话
                group_id: str = "", # 给群AI用的id
                bot_name = "", # 群AI的自己认可的名字
                model: str = "ernie-bot-turbo", # 详见https://github.com/PaddlePaddle/ERNIE-Bot-SDK/blob/develop/docs/models.md
                temperature: Number = 0.95, # 用于调节模型输出概率分布，从而控制生成结果的随机性。 默认0.95，范围为(0, 1.0]，不能为0；
                # max_tokens: int = 1000, # ERNIE没有这个选项，默认3000，ernie-bot-8k是7000
                penalty_score: Number = 1.0, # 通过对已生成的token增加惩罚，减少重复生成的现象。此参数值越高则惩罚越大。值越大表示惩罚越大；默认1.0，取值范围：[1.0, 2.0]。
                # frequency_penalty: Number = 0, # ERNIE没有这个
                group_context_max: int = 3, # 记录群消息，临时会话不需要这个
                voice: bool = False,
                system: str = ""  # ERNIE需要单独传设定
    ):
        if qq:
        # 如果是临时会话则不添加预设定
            self.system = ""
        else:
        # 如果要加其他设定可以通过 <添加群设定> 添加
            self.system =f"""你的名字是“{bot_name}”，你应该假装自己是一个人类，我会提供一些聊天记录，就好像这样：
摸余：早
袋鼠：早什么早,听好，早上要说我爱你
摸余：啊这
这里“摸余”和“袋鼠”都是群昵称。你要当作自己就在群里，并且恰当地加入会话。
如果不知道说什么，可以分析一下群友的心理"""
        self.messages = []
        self.conversation_id = conversation_id
        self.qq = qq
        self.group_id = group_id
        self.model = model
        self.bot_name = bot_name
        self.temperature = temperature
        self.penalty_score = penalty_score
        self.group_context_max = group_context_max
        if group_context_max == -1:
            self.group_context = deque([])
        else:
            self.group_context = deque([], group_context_max)  # erniebot.errors.InvalidArgumentError: `messages` must have an odd number of elements. 必须要求奇数
        self.full_token_cost = 0
        self.last_token_cost = 0
        self.voice = voice

    def add_group_context(self, role, msg):
        message = {"role": role, "content": msg}
        self.group_context.append(message)

    def get_group_reply(self, msg: str):
    # for group_AI use, message will be added to self.group_context
        if self.group_context_max == 0:
            return self.get_reply(msg)
        try:
            response = self.get_full_response(self.messages + list(self.group_context))
            reply = response.get_result().strip()
            # reply = re.sub(r'@(\S+)', '', reply)
            self.add_group_context("assistant", reply)
            token_cost = response["usage"]["total_tokens"]
            self.last_token_cost = token_cost
            self.full_token_cost += token_cost
            return reply
        #except openai.error.OpenAIError as e:
        except erniebot.errors.EBError as e:
            # print(e.http_body['type'])
            try:
                return f"error {e.http_status}: {e.http_body['type']}"
            except:
                return str(traceback.format_exc())

    def add_conversation_msg(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.messages.append(message)

    def get_full_response(self, messages):
        # 这里需要做点修改，如果模型中不包含'vision'。则需要整理一下content，移除image_url
        # 直接用list中第一个dict的text替换整个list。
        # ERNIE好像暂时没有支持图片输入的模型，这里先放着不动了
        if "vision" not in self.model:
            for message in messages:
                if message['role'] == "user" and type(message['content']) == list:
                    message['content'] = message['content'][0]['text']
        response = erniebot.ChatCompletion.create(
            model=self.model,
            messages=messages,
            system=self.system,
            temperature=self.temperature,  # 用于调节模型输出概率分布，从而控制生成结果的随机性。(1) 较高的数值会使生成结果更加随机，而较低的数值会使结果更加集中和确定；(2) 默认0.95，范围为(0, 1.0]，不能为0；(3) 建议只设置此参数和top_p中的一个。
            penalty_score=self.penalty_score, #default 0. between -2.0 and 2.0, increasing the model's likelihood to talk about new topics
        )
        return response
    
    def get_reply(self, msg: str):
    # for temp_chat use, message will be added to self.messages
        self.add_conversation_msg("user", msg)
        try:
            response = self.get_full_response(self.messages)
            reply = response.get_result().strip()
            self.add_conversation_msg("assistant", reply)
            token_cost = response["usage"]["total_tokens"]
            self.last_token_cost = token_cost
            self.full_token_cost += token_cost
            return reply
        except erniebot.errors.EBError as e:
            try:
                return f"error {e.http_status}: {e.http_body['type']}"
            except:
                return str(e.http_body)
            return e._message

    def get_system_inputs(self):
    # get existing system inputs
        return self.system

    def add_conversation_setting(self, msg: str):
        self.system = msg

    def clear_all(self):
        self.messages.clear()
        self.group_context.clear()
        self.full_token_cost = 0
        self.last_token_cost = 0

    def clear_messages(self):
        new_messages = []
        self.group_context.clear()
        self.full_token_cost = 0
        self.last_token_cost = 0
        self.messages = new_messages

    def get_full_token_cost(self):
        return self.full_token_cost

    def get_last_token_cost(self):
        return self.last_token_cost

    def get_conversation_id(self):
        return self.conversation_id

    def to_dict(self):
        output = {
            "messages": self.messages,
            "conversation_id": self.conversation_id,
            "qq": self.qq,
            "group_id": self.group_id,
            "model": self.model,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
            "group_context": list(self.group_context),
            "group_context_max": self.group_context_max,
            "full_token_cost": self.full_token_cost,
            "last_token_cost": self.last_token_cost,
            "voice": self.voice
        }
        return output

    def load_dict(self, conversation: dict):
        self.messages = conversation["messages"]
        self.conversation_id = conversation["conversation_id"]
        self.qq = conversation["qq"]
        self.group_id = conversation["group_id"]
        self.model = conversation["model"]
        self.temperature = conversation["temperature"]
        self.penalty_score = conversation["penalty_score"]
        self.group_context_max = conversation["group_context_max"]
        if self.group_context_max == -1:
            self.group_context = deque([])
        else:
            self.group_context = deque([], self.group_context_max)  # erniebot.errors.InvalidArgumentError: `messages` must have an odd number of elements. 必须要求奇数
        self.group_context.extend(conversation["group_context"])
        self.full_token_cost = conversation["full_token_cost"]
        self.last_token_cost = conversation["last_token_cost"]
        try:
            self.voice = conversation["voice"]
        except:
            conversation["voice"] = False
            self.voice = False