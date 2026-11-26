# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import asyncio
import base64
import functools
import json
import re
import time
import traceback

import httpx
from httpx import Timeout

LLM_LIST = ["doubao", "gpt", "qwen", "internvl"]


def log_time(func):
    """
    装饰器：打印方法的执行耗时和方法名称
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"方法 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
        return result

    return wrapper


class LlmUtil:
    def __init__(self, sys_content="You are a helpful assistant.", raise_error=True):
        self.agent_url = "https://one-api.sensoro.com/v1/chat/completions"
        self.agent_headers = {
            "Authorization":"Bearer sk-DpnzsrnqfZTjmicf48011f2b6e1e4bBf9d22A5Ca7301505b"
        }
        self.qwen_url = "https://qwn-llm.xn.sensoro.vip" + "/v1/chat/completions"
        self.qwen_model = "/work/Qwen2.5-VL-72B-Instruct-Pointer-AWQ"
        self.intern_vl_url = (
            "https://internvl-llm.xn.sensoro.vip" + "/v1/chat/completions"
        )
        self.intern_vl_model = "internvl-internlm2"
        self.doubao_model = "doubao-pro-vision-2"
        # self.gpt_model = "sensoro-gpt"  # 微软
        self.gpt_model = "sensoro-gptv2"  # openai: 效果差
        # self.gpt_model = "gpt-4o-2024-08-06-foundry"  # openai: 压测
        self.retry_times, self.timeout = 3, 180
        self.sys_content = sys_content
        self.raise_error = raise_error

    @staticmethod
    async def url2base64(url, with_content=False):
        async with httpx.AsyncClient(timeout=Timeout(timeout=60)) as client:
            r = await client.get(url)
            assert r.status_code == 200, f"{r.status_code}, {r.text}"
            if with_content:
                return base64.b64encode(r.content).decode(), r.content
            return base64.b64encode(r.content).decode()

    @staticmethod
    def path2base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def req(self, url, body, headers=None):
        for i in range(3):
            try:
                async with httpx.AsyncClient(
                    timeout=Timeout(timeout=self.timeout)
                ) as client:
                    r = await client.post(url, json=body, headers=headers or {})
                    assert r.status_code == 200, f"{r.status_code}, {r.text}"
                    return r.json()
            except Exception as e:
                print(f"req[{url}] error: {traceback.format_exc()}\n{e}")
                await asyncio.sleep(0.1)
        if self.raise_error:
            raise Exception(f"req[{url}] error")

    def gen_body_msg(self, prompt, image_base64_data):
        image_base64_list = (
            image_base64_data
            if isinstance(image_base64_data, list)
            else [image_base64_data]
        )
        return [
            {"role": "system", "content": self.sys_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                    for image_base64 in image_base64_list
                ],
            },
        ]

    def gen_body_msg_noimg(self, prompt):
        return [
            {"role": "system", "content": self.sys_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]


    async def call_agent(self, model, content, image_base64_data):
        body = {
            "model": model,
            "temperature": 0,
            "messages": self.gen_body_msg(content, image_base64_data),
        }
        resp = await self.req(self.agent_url, body, self.agent_headers)
        if resp is None:
            return
        return resp["choices"][0]["message"]["content"].strip()

    async def call_agent_noimg(self, model, content):
        body = {
            "model": model,
            "temperature": 0,
            "messages": self.gen_body_msg_noimg(content),
        }
        resp = await self.req(self.agent_url, body, self.agent_headers)
        if resp is None:
            return
        return resp["choices"][0]["message"]["content"].strip()


    # @log_time
    async def call_doubao(self, content, image_base64_data):
        return await self.call_agent(self.doubao_model, content, image_base64_data)

    # @log_time
    async def call_gpt(self, content, image_base64_data):
        return await self.call_agent(self.gpt_model, content, image_base64_data)


    # @log_time
    async def call_doubao_noimg(self, content):
        return await self.call_agent_noimg(self.doubao_model, content)

    # @log_time
    async def call_gpt_noimg(self, content):
        return await self.call_agent_noimg(self.gpt_model, content)





    @staticmethod
    def parse_content(content):
        if "```json" in content:
            content_data = re.findall(r"```json(.*?)```", content, re.S)
            content_data = json.loads(content_data[0])
        else:
            try:
                content_data = json.loads(content)
            except json.decoder.JSONDecodeError:
                content_data = str(content)
            except Exception as e:
                content_data = str(content)
        return content_data

    # @log_time
    async def call_qwen(self, content, image_base64_data):
        body = {
            "model": self.qwen_model,
            "temperature": 0,
            "messages": self.gen_body_msg(content, image_base64_data),
        }
        resp = await self.req(self.qwen_url, body)
        return resp["choices"][0]["message"]["content"]

    # @log_time
    async def call_qwen_noimg(self, content):
        body = {
            "model": self.qwen_model,
            "temperature": 0,
            "messages": self.gen_body_msg_noimg(content),
        }
        resp = await self.req(self.qwen_url, body)
        return resp["choices"][0]["message"]["content"]

        
        

    # @log_time
    async def call_internvl(self, content, image_base64_data):
        body = {
            "model": self.intern_vl_model,
            "temperature": 0,
            "messages": self.gen_body_msg(content, image_base64_data),
        }
        resp = await self.req(self.intern_vl_url, body)
        return resp["choices"][0]["message"]["content"]

    # @log_time
    async def call_internvl_noimg(self, content):
        body = {
            "model": self.intern_vl_model,
            "temperature": 0,
            "messages": self.gen_body_msg_noimg(content),
        }
        resp = await self.req(self.intern_vl_url, body)
        return resp["choices"][0]["message"]["content"]





llm_util = LlmUtil(sys_content="你是一位专业图像标注员")