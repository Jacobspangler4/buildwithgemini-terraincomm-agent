# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.code_executors import AgentEngineSandboxCodeExecutor
from google.adk.models import Gemini
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from a2ui.basic_catalog.provider import BasicCatalog
from a2ui.schema.manager import A2uiSchemaManager

from .a2ui_utils import a2ui_callback
from .tools import (
    calculate_fspl_and_rssi,
    calculate_fresnel_zone,
    calculate_wall_attenuation,
    optimize_ap_placement,
    calculate_link_budget_and_snr,
    fetch_terrain_elevation_profile,
    consult_corpus,
    generate_terrain_rf_image,
)

MODEL = "gemini-3.6-flash"
AGENT_ENGINE_RESOURCE_NAME = "projects/432975831710/locations/us-east1/reasoningEngines/5082259402929471488"

a2ui_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[BasicCatalog.get_config("0.8")],
)

RF_ENGINEER_INSTRUCTION = a2ui_manager.generate_system_prompt(
    role_description=(
        "You are an expert Radio Frequency (RF) Engineer and Wireless Network Architect. "
        "Your primary mission is to assist network engineers, system integrators, and IT teams "
        "in designing, modeling, and optimizing wireless network connections, access point (AP) placements, "
        "and point-to-point RF links."
    ),
    workflow_description=(
        "Analyze the request and return structured UI when appropriate. Core Capabilities & Tools available to you: "
        "1. `calculate_fspl_and_rssi`: Compute Free Space Path Loss (FSPL), Effective Isotropic Radiated Power (EIRP), and Received Signal Strength (RSSI). "
        "2. `calculate_fresnel_zone`: Calculate 1st Fresnel zone radius, 60% line-of-sight (LoS) clearance threshold, and assess physical obstruction risks. "
        "3. `calculate_wall_attenuation`: Estimate signal loss (dB) across 2.4GHz, 5GHz, and 6GHz through various wall partitions (drywall, glass, concrete, brick, metal doors). "
        "4. `optimize_ap_placement`: Generate optimal Access Point count, (X, Y) layout grid coordinates, effective coverage radius, and non-overlapping channel plans (1/6/11 for 2.4GHz, UNII channels for 5GHz/6GHz) for rectangular floor plans. "
        "5. `calculate_link_budget_and_snr`: Determine Signal-to-Noise Ratio (SNR), link margin, maximum estimated MCS index, and application suitability (VoIP, 4K Streaming, AR/VR). "
        "6. `fetch_terrain_elevation_profile`: Fetch or model ground elevation profile points (meters AMSL) along a line-of-sight path between two geographic coordinates. "
        "7. `consult_corpus`: Search and retrieve grounded passages from the imported Project Gutenberg document corpus. "
        "8. `generate_terrain_rf_image`: Generate 2D visual maps, antenna placement diagrams, or terrain propagation diagrams using gemini-3.1-flash-lite-image. "
        "9. Python Code Sandbox Execution: Execute Python code in a safe Agent Engine Sandbox environment for complex calculations. "
        "Memory: You remember ALL user transmitter locations across sessions. Explicitly acknowledge when storing site parameters."
    ),
    ui_description=(
        "Keep every surface tiny and flat: ONE Card > ONE Column > a few Text rows. "
        "Never nest a Card inside a Card. "
        "Use ONLY these components: Card, Column, Row, Text, and Image. Do not use "
        "Table or Heading (unsupported), or Buttons, actions, or forms (they do "
        "nothing in adk web). "
        "You may include one Image component, but only when you have a public https "
        "URL for the image (for example the URL an image tool returns after uploading "
        "to a public bucket). Set the Image url to that exact https link, for example "
        '{"Image": {"url": {"literalString": "https://..."}}}. Never point an '
        "Image at a bare filename, an artifact name, or a non-http(s) path. If you do "
        "not have a public URL, add a short Text line noting the image instead. "
        "No markdown in text; use the usageHint property ('h1', 'h2', 'body') for "
        "headings and emphasis. "
        "Output ONLY the raw A2UI JSON array — no prose, and never wrap it in "
        "<a2a_datapart_json> tags or 'kind'/'data'/'metadata' objects."
    ),
    include_schema=True,
    include_examples=True,
)


async def generate_memories_callback(callback_context: CallbackContext):
    """Callback after each turn to extract and persist durable memories to Vertex AI Memory Bank."""
    await callback_context.add_session_to_memory()
    return None


root_agent = Agent(
    name="terraincomm_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=RF_ENGINEER_INSTRUCTION,
    tools=[
        PreloadMemoryTool(),
        calculate_fspl_and_rssi,
        calculate_fresnel_zone,
        calculate_wall_attenuation,
        optimize_ap_placement,
        calculate_link_budget_and_snr,
        fetch_terrain_elevation_profile,
        consult_corpus,
        generate_terrain_rf_image,
    ],
    code_executor=AgentEngineSandboxCodeExecutor(
        agent_engine_resource_name=AGENT_ENGINE_RESOURCE_NAME
    ),
    after_model_callback=a2ui_callback,
    after_agent_callback=generate_memories_callback,
)






app = App(
    root_agent=root_agent,
    name="app",
)

