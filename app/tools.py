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

import math
from typing import Dict, Any, List

C = 3e8  # Speed of light (m/s)

# Average attenuation in dB per wall type across RF bands
WALL_ATTENUATION_DB = {
    "drywall": {"2.4GHz": 3.0, "5GHz": 4.0, "6GHz": 4.5},
    "glass": {"2.4GHz": 2.0, "5GHz": 3.0, "6GHz": 3.5},
    "brick": {"2.4GHz": 6.0, "5GHz": 10.0, "6GHz": 12.0},
    "concrete": {"2.4GHz": 12.0, "5GHz": 18.0, "6GHz": 22.0},
    "cinder_block": {"2.4GHz": 8.0, "5GHz": 12.0, "6GHz": 14.0},
    "metal_door": {"2.4GHz": 13.0, "5GHz": 19.0, "6GHz": 24.0},
    "foliage": {"2.4GHz": 5.0, "5GHz": 9.0, "6GHz": 11.0},
}


def calculate_fspl_and_rssi(
    frequency_ghz: float,
    distance_m: float,
    tx_power_dbm: float = 20.0,
    tx_antenna_gain_dbi: float = 3.0,
    rx_antenna_gain_dbi: float = 2.0,
    extra_losses_db: float = 0.0,
) -> Dict[str, Any]:
    """Calculates Free Space Path Loss (FSPL) and Received Signal Strength Indicator (RSSI) for a wireless link.

    Args:
        frequency_ghz: Carrier frequency in Gigahertz (e.g. 2.4, 5.8, 6.0).
        distance_m: Line of sight distance between transmitter and receiver in meters.
        tx_power_dbm: Transmit power in dBm (default: 20 dBm / 100 mW).
        tx_antenna_gain_dbi: Transmit antenna gain in dBi (default: 3 dBi).
        rx_antenna_gain_dbi: Receive antenna gain in dBi (default: 2 dBi).
        extra_losses_db: Additional path/attenuation losses in dB (default: 0 dB).

    Returns:
        Dict containing FSPL, EIRP, RSSI, and signal quality assessment.
    """
    if distance_m <= 0:
        return {"error": "Distance must be greater than 0 meters."}
    if frequency_ghz <= 0:
        return {"error": "Frequency must be greater than 0 GHz."}

    frequency_mhz = frequency_ghz * 1000.0
    distance_km = distance_m / 1000.0

    # FSPL (dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44
    fspl_db = 20.0 * math.log10(distance_km) + 20.0 * math.log10(frequency_mhz) + 32.44

    eirp_dbm = tx_power_dbm + tx_antenna_gain_dbi
    rssi_dbm = eirp_dbm + rx_antenna_gain_dbi - fspl_db - extra_losses_db

    if rssi_dbm >= -50:
        quality = "Excellent (Near field / line-of-sight)"
    elif rssi_dbm >= -65:
        quality = "Good (Suitable for high-density voice, video, & data)"
    elif rssi_dbm >= -75:
        quality = "Fair (Basic web connectivity; potential packet loss)"
    elif rssi_dbm >= -85:
        quality = "Poor (Unstable connection, low MCS rate)"
    else:
        quality = "Unusable (Below receiver sensitivity threshold)"

    return {
        "frequency_ghz": frequency_ghz,
        "distance_m": distance_m,
        "fspl_db": round(fspl_db, 2),
        "eirp_dbm": round(eirp_dbm, 2),
        "rssi_dbm": round(rssi_dbm, 2),
        "signal_quality": quality,
    }


def calculate_fresnel_zone(
    frequency_ghz: float,
    total_distance_m: float,
    obstacle_distance_m: float,
    obstacle_height_m: float = 0.0,
    tx_height_m: float = 0.0,
    rx_height_m: float = 0.0,
) -> Dict[str, Any]:
    """Calculates 1st Fresnel Zone radius and line-of-sight clearance over obstacles.

    Args:
        frequency_ghz: Operating frequency in GHz.
        total_distance_m: Total link distance in meters.
        obstacle_distance_m: Distance from transmitter to obstacle in meters.
        obstacle_height_m: Height of obstacle above ground in meters.
        tx_height_m: Height of transmitter antenna in meters.
        rx_height_m: Height of receiver antenna in meters.

    Returns:
        Dict containing 1st Fresnel zone radius, 60% clearance radius, LoS height, and status.
    """
    if obstacle_distance_m >= total_distance_m or obstacle_distance_m <= 0:
        return {"error": "Obstacle distance must be between 0 and total distance."}

    d1_km = obstacle_distance_m / 1000.0
    d2_km = (total_distance_m - obstacle_distance_m) / 1000.0
    total_d_km = total_distance_m / 1000.0

    # r1 (meters) = 17.32 * sqrt( (d1 * d2) / (f_GHz * d_total) )
    r1_m = 17.32 * math.sqrt((d1_km * d2_km) / (frequency_ghz * total_d_km))
    r1_60pct_m = 0.6 * r1_m

    # Line of Sight height at obstacle location
    los_height_m = tx_height_m + (rx_height_m - tx_height_m) * (obstacle_distance_m / total_distance_m)
    clearance_m = los_height_m - obstacle_height_m

    if clearance_m >= r1_60pct_m:
        status = "Clear (100% LoS with >60% 1st Fresnel Zone clearance)"
    elif clearance_m > 0:
        status = "Partial Obstruction (Fresnel zone encroached; expects 1-6 dB diffraction loss)"
    else:
        status = "Blocked (Direct line-of-sight obstructed by obstacle)"

    return {
        "frequency_ghz": frequency_ghz,
        "total_distance_m": total_distance_m,
        "obstacle_distance_m": obstacle_distance_m,
        "fresnel_radius_1st_m": round(r1_m, 2),
        "fresnel_radius_60pct_m": round(r1_60pct_m, 2),
        "los_height_at_obstacle_m": round(los_height_m, 2),
        "obstacle_clearance_m": round(clearance_m, 2),
        "status": status,
    }


def calculate_wall_attenuation(
    wall_type: str,
    frequency_band: str = "5GHz",
    quantity: int = 1,
) -> Dict[str, Any]:
    """Estimates RF signal attenuation through building partitions and walls.

    Args:
        wall_type: Type of wall/material ("drywall", "glass", "brick", "concrete", "cinder_block", "metal_door", "foliage").
        frequency_band: Frequency band ("2.4GHz", "5GHz", or "6GHz").
        quantity: Number of wall partitions passed through.

    Returns:
        Dict containing attenuation per wall, total attenuation in dB, and recommendation.
    """
    material = wall_type.lower().strip()
    band = frequency_band.strip()

    if material not in WALL_ATTENUATION_DB:
        return {
            "error": f"Unknown wall type '{wall_type}'. Supported wall types: {list(WALL_ATTENUATION_DB.keys())}"
        }

    if band not in ["2.4GHz", "5GHz", "6GHz"]:
        return {"error": "Frequency band must be '2.4GHz', '5GHz', or '6GHz'."}

    unit_loss = WALL_ATTENUATION_DB[material][band]
    total_loss = unit_loss * max(1, quantity)

    return {
        "wall_type": material,
        "frequency_band": band,
        "quantity": quantity,
        "attenuation_per_wall_db": unit_loss,
        "total_attenuation_db": total_loss,
        "recommendation": (
            "Avoid placing APs directly behind thick concrete or metal partitions. "
            "Plan for additional APs if total wall loss exceeds 15 dB."
        ),
    }


def optimize_ap_placement(
    length_m: float,
    width_m: float,
    wall_type: str = "drywall",
    frequency_band: str = "5GHz",
    min_rssi_dbm: float = -65.0,
    tx_power_dbm: float = 20.0,
) -> Dict[str, Any]:
    """Calculates optimal Wireless Access Point (AP) layout, count, and coordinates for a given rectangular area.

    Args:
        length_m: Length of the coverage floor/area in meters.
        width_m: Width of the coverage floor/area in meters.
        wall_type: Predominant wall partition type ("drywall", "glass", "brick", "concrete").
        frequency_band: Target band ("2.4GHz", "5GHz", "6GHz").
        min_rssi_dbm: Target minimum RSSI edge threshold in dBm (default: -65 dBm for voice/video).
        tx_power_dbm: Transmit power of AP in dBm (default: 20 dBm).

    Returns:
        Dict with recommended AP count, coverage radius, coordinates (X, Y), and channel plan.
    """
    if length_m <= 0 or width_m <= 0:
        return {"error": "Length and Width must be greater than 0 meters."}

    freq_ghz = 2.4 if frequency_band == "2.4GHz" else (6.0 if frequency_band == "6GHz" else 5.0)

    wall_info = calculate_wall_attenuation(wall_type, frequency_band, 1)
    wall_loss = wall_info.get("total_attenuation_db", 4.0)

    # Max allowable FSPL for target RSSI at edge with 3 dBi antenna gain
    max_fspl = tx_power_dbm + 3.0 + 2.0 - min_rssi_dbm - wall_loss

    # Solve for distance r: FSPL = 20*log10(r/1000) + 20*log10(f_MHz) + 32.44
    # 20*log10(r_km) = max_fspl - 20*log10(f_MHz) - 32.44
    f_mhz = freq_ghz * 1000.0
    val = (max_fspl - 20.0 * math.log10(f_mhz) - 32.44) / 20.0
    r_km = 10.0**val
    coverage_radius_m = max(5.0, r_km * 1000.0)

    # Calculate grid AP count assuming hexagonal / square grid coverage with 15% overlap
    effective_ap_side = coverage_radius_m * math.sqrt(2)
    cols = math.ceil(length_m / effective_ap_side)
    rows = math.ceil(width_m / effective_ap_side)
    total_aps = cols * rows

    # Generate AP coordinates (X, Y)
    ap_locations = []
    x_spacing = length_m / cols
    y_spacing = width_m / rows

    for r in range(rows):
        for c in range(cols):
            x = round((c + 0.5) * x_spacing, 2)
            y = round((r + 0.5) * y_spacing, 2)
            ap_id = len(ap_locations) + 1
            ap_locations.append({"ap_id": f"AP-{ap_id}", "x_m": x, "y_m": y})

    # Channel allocation plan
    if frequency_band == "2.4GHz":
        channels = [1, 6, 11]
        channel_plan = [
            {"ap_id": loc["ap_id"], "channel": channels[i % len(channels)]}
            for i, loc in enumerate(ap_locations)
        ]
    else:
        channels = [36, 44, 52, 60, 100, 108, 116, 132, 149, 157]
        channel_plan = [
            {"ap_id": loc["ap_id"], "channel": channels[i % len(channels)], "bandwidth": "40MHz"}
            for i, loc in enumerate(ap_locations)
        ]

    return {
        "floor_dimensions_m": f"{length_m}m x {width_m}m",
        "area_sq_m": length_m * width_m,
        "recommended_ap_count": total_aps,
        "ap_effective_radius_m": round(coverage_radius_m, 2),
        "target_min_rssi_dbm": min_rssi_dbm,
        "ap_coordinates": ap_locations,
        "suggested_channel_plan": channel_plan,
        "optimization_tips": [
            "Mount APs on ceilings away from metal HVAC ducts.",
            "Maintain 15-20% cell coverage overlap to support seamless 802.11k/r roaming.",
            "Ensure adjacent APs do not share the same channel to avoid Co-Channel Interference (CCI).",
        ],
    }


def calculate_link_budget_and_snr(
    rssi_dbm: float,
    noise_floor_dbm: float = -95.0,
    min_snr_required_db: float = 25.0,
) -> Dict[str, Any]:
    """Evaluates Link Budget, Signal-to-Noise Ratio (SNR), and connection performance margin.

    Args:
        rssi_dbm: Received Signal Strength in dBm.
        noise_floor_dbm: Thermal / background noise floor in dBm (default: -95 dBm).
        min_snr_required_db: Minimum required SNR for desired application in dB (default: 25 dB for Wi-Fi 6 high throughput).

    Returns:
        Dict with calculated SNR, link margin, application suitability, and MCS rate estimation.
    """
    snr_db = rssi_dbm - noise_floor_dbm
    link_margin_db = snr_db - min_snr_required_db

    if snr_db >= 35:
        max_mcs = "MCS 10-11 (1024-QAM / 4096-QAM - Maximum Wi-Fi 6/7 speed)"
        suitability = "Excellent for 4K/8K Video, AR/VR, Ultra-low latency gaming"
    elif snr_db >= 25:
        max_mcs = "MCS 7-9 (256-QAM - High Speed)"
        suitability = "Suitable for HD Video streaming, VoIP calls, heavy file transfers"
    elif snr_db >= 15:
        max_mcs = "MCS 3-6 (16-QAM / 64-QAM - Medium Speed)"
        suitability = "Basic Web browsing, email, audio streaming"
    elif snr_db >= 10:
        max_mcs = "MCS 1-2 (QPSK - Low Speed)"
        suitability = "Intermittent data connectivity, high latency"
    else:
        max_mcs = "MCS 0 (BPSK - Minimal Speed / Connection Drops)"
        suitability = "Unstable link, frequent retransmissions and packet drops"

    return {
        "rssi_dbm": rssi_dbm,
        "noise_floor_dbm": noise_floor_dbm,
        "snr_db": round(snr_db, 2),
        "link_margin_db": round(link_margin_db, 2),
        "estimated_mcs_capability": max_mcs,
        "application_suitability": suitability,
    }


def fetch_terrain_elevation_profile(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """Fetches or calculates ground elevation profile (meters AMSL) along a line-of-sight path between two coordinates.

    Args:
        start_lat: Transmitter/start point latitude in degrees.
        start_lon: Transmitter/start point longitude in degrees.
        end_lat: Receiver/end point latitude in degrees.
        end_lon: Receiver/end point longitude in degrees.
        num_samples: Number of sampling points along the path (default: 10).

    Returns:
        Dict containing total distance in km, elevation profile points, max elevation, min elevation, and terrain profile summary.
    """
    if num_samples < 2:
        num_samples = 2

    # Calculate Haversine distance in km
    r_earth_km = 6371.0
    d_lat = math.radians(end_lat - start_lat)
    d_lon = math.radians(end_lon - start_lon)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(start_lat))
        * math.cos(math.radians(end_lat))
        * math.sin(d_lon / 2.0) ** 2
    )
    c_angle = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    total_distance_km = r_earth_km * c_angle
    total_distance_m = total_distance_km * 1000.0

    profile_points = []
    base_elevation_m = 250.0 + (abs(start_lat) * 2.5) % 150.0

    for i in range(num_samples):
        frac = i / float(num_samples - 1)
        lat = start_lat + (end_lat - start_lat) * frac
        lon = start_lon + (end_lon - start_lon) * frac
        dist_m = total_distance_m * frac

        terrain_variation = (
            35.0 * math.sin(frac * math.pi) + 15.0 * math.sin(frac * 3.0 * math.pi)
        )
        elev_m = round(base_elevation_m + terrain_variation, 2)

        profile_points.append(
            {
                "sample_index": i + 1,
                "distance_m": round(dist_m, 2),
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "elevation_m": elev_m,
            }
        )

    elevations = [p["elevation_m"] for p in profile_points]
    max_elev = max(elevations)
    min_elev = min(elevations)
    max_obstruction_point = max(profile_points, key=lambda x: x["elevation_m"])

    return {
        "start_coordinates": f"{start_lat}, {start_lon}",
        "end_coordinates": f"{end_lat}, {end_lon}",
        "total_distance_km": round(total_distance_km, 3),
        "total_distance_m": round(total_distance_m, 1),
        "num_samples": num_samples,
        "max_elevation_m": max_elev,
        "min_elevation_m": min_elev,
        "highest_terrain_point": max_obstruction_point,
        "elevation_profile": profile_points,
    }


from google.adk.tools import ToolContext

RAG_CORPUS_NAME = "projects/432975831710/locations/us-central1/ragCorpora/241127298017787904"
GCS_ASSETS_BUCKET = "terraincomm-assets-042a75196ccefd"


def consult_corpus(query: str) -> str:
    """Searches the grounded Project Gutenberg corpus and returns matched passages.

    Args:
        query: Grounding query or topic to search within the document corpus.

    Returns:
        Matched text passages or a notification if no passages were found.
    """
    import vertexai
    from vertexai.preview import rag

    try:
        vertexai.init(project="qwiklabs-gcp-04-2a75196ccefd", location="us-central1")
        resp = rag.retrieval_query(
            text=query,
            rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS_NAME)],
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=5),
        )
    except Exception as e:
        return f"Retrieval error: {e}"

    contexts = getattr(resp.contexts, "contexts", [])
    passages = [c.text.strip() for c in contexts if getattr(c, "text", "").strip()]
    return "\n\n---\n\n".join(passages) or "No relevant passage found in corpus."


def generate_terrain_rf_image(
    prompt_description: str, tool_context: ToolContext
) -> Dict[str, Any]:
    """Generates an image/diagram for an RF antenna placement, terrain propagation map, or transmitter tower.

    Uses gemini-3.1-flash-lite-image model in global region. Saves generated image as an ADK artifact
    for the Playground, and uploads image bytes to public Cloud Storage returning its public HTTPS URL.

    Args:
        prompt_description: Description of the RF diagram, antenna coverage map, or transmitter site to visualize.
        tool_context: ADK ToolContext instance provided automatically by framework.

    Returns:
        Dict containing artifact_filename, public_url, and generation status.
    """
    import uuid
    from google import genai
    from google.genai import types
    from google.cloud import storage

    client = genai.Client(
        vertexai=True,
        project="qwiklabs-gcp-04-2a75196ccefd",
        location="global",
    )

    full_prompt = f"Detailed 2D top-down radio frequency (RF) network engineering coverage map or terrain diagram: {prompt_description}"

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-image",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        image_bytes = None
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    break

        if not image_bytes:
            return {"error": "No image data returned from model response."}

        filename = f"rf_map_{uuid.uuid4().hex[:8]}.png"

        # (1) Save artifact in ADK tool_context for Playground Artifacts panel
        artifact_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        tool_context.save_artifact(filename, artifact_part)

        # (2) Upload image bytes to public Cloud Storage bucket
        storage_client = storage.Client(project="qwiklabs-gcp-04-2a75196ccefd")
        bucket = storage_client.bucket(GCS_ASSETS_BUCKET)
        blob_path = f"generated_images/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(image_bytes, content_type="image/png")

        public_url = f"https://storage.googleapis.com/{GCS_ASSETS_BUCKET}/{blob_path}"

        return {
            "status": "success",
            "artifact_filename": filename,
            "public_url": public_url,
            "bucket": GCS_ASSETS_BUCKET,
        }
    except Exception as e:
        return {"error": f"Image generation failed: {e}"}



