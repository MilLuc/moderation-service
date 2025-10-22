import asyncio
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager

import torch

# Imports required by the service's model
from common_code.common.enums import (
    ExecutionUnitTagAcronym,
    ExecutionUnitTagName,
    FieldDescriptionType,
)
from common_code.common.models import ExecutionUnitTag, FieldDescription
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import Logger, get_logger
from common_code.service.controller import router as service_router
from common_code.service.enums import ServiceStatus
from common_code.service.models import Service
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.models import TaskData
from common_code.tasks.service import TasksService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from models import ModerationInput, ModerationResult, Risk

INPUT_KEY = "input"
RESULT_KEY = "result"
RISK_NAME_KEY = "risk_name"

settings = get_settings()

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

model_path_name = "ibm-granite/granite-guardian-3.2-5b"

safe_token = "No"
risky_token = "Yes"
nlogprobs = 20

tokenizer = AutoTokenizer.from_pretrained(model_path_name)

sampling_params = SamplingParams(temperature=0.0, logprobs=nlogprobs)
model = LLM(model=model_path_name, tensor_parallel_size=1)


def get_probabilities(logprobs):
    safe_token_prob = 1e-50
    risky_token_prob = 1e-50
    for gen_token_i in logprobs:
        for token_prob in gen_token_i.values():
            decoded_token = token_prob.decoded_token
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(token_prob.logprob)
            if decoded_token.strip().lower() == risky_token.lower():
                risky_token_prob += math.exp(token_prob.logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(risky_token_prob)]), dim=0
    )

    return probabilities


def parse_output(output):
    label, prob_of_risk = None, None

    if nlogprobs > 0:
        logprobs = next(iter(output.outputs)).logprobs
        if logprobs is not None:
            prob = get_probabilities(logprobs)
            prob_of_risk = prob[1]

    output = next(iter(output.outputs)).text.strip()
    res = re.search(r"^\w+", output, re.MULTILINE).group(0).strip()
    if risky_token.lower() == res.lower():
        label = risky_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"

    confidence_level = (
        re.search(r"<confidence> (.*?) </confidence>", output).group(1).strip()
    )

    return label, confidence_level, prob_of_risk.item()


def detect_risk(risk: Risk, messages):
    guardian_config = {RISK_NAME_KEY: risk}
    chat = tokenizer.apply_chat_template(
        messages,
        guardian_config=guardian_config,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = model.generate(chat, sampling_params, use_tqdm=False)
    predicted_label = output[0].outputs[0].text.strip()

    label, confidence, prob = parse_output(predicted_label)
    return ModerationResult(
        risk_name=risk,
        detected=label == risky_token,
        confidence=confidence,
        probability=prob,
    )


class MyService(Service):
    """
    Moderation service
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Moderation Service",
            slug="moderation-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name=INPUT_KEY,
                    type=[
                        FieldDescriptionType.APPLICATION_JSON,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name=RESULT_KEY, type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.GENERIC,
                    acronym=ExecutionUnitTagAcronym.GENERIC,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        input = ModerationInput(**data[INPUT_KEY])
        messages = input.to_messages()
        result = []
        for risk in input.risks_to_detect:
            result.append[detect_risk(risk, messages)]
        return {
            RESULT_KEY: TaskData(
                data=json.dump(result), type=FieldDescriptionType.APPLICATION_JSON
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """
This Service validates the given prompt based on various guards
"""
api_summary = """
A
"""

# Define the FastAPI application with information
# TODO: 7. CHANGE THE API TITLE, VERSION, CONTACT AND LICENSE
app = FastAPI(
    lifespan=lifespan,
    title="Sample Service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
