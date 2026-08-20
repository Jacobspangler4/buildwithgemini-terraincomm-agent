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

from app.tools import (
    calculate_fspl_and_rssi,
    calculate_fresnel_zone,
    calculate_wall_attenuation,
    optimize_ap_placement,
    calculate_link_budget_and_snr,
)


def test_calculate_fspl_and_rssi():
    res = calculate_fspl_and_rssi(frequency_ghz=5.0, distance_m=100.0, tx_power_dbm=20.0)
    assert "fspl_db" in res
    assert "rssi_dbm" in res
    assert res["frequency_ghz"] == 5.0
    assert res["distance_m"] == 100.0
    # FSPL for 5 GHz at 100m is approx 86.4 dB
    assert 80.0 < res["fspl_db"] < 92.0


def test_calculate_fresnel_zone():
    res = calculate_fresnel_zone(
        frequency_ghz=5.8,
        total_distance_m=1000.0,
        obstacle_distance_m=500.0,
        obstacle_height_m=5.0,
        tx_height_m=10.0,
        rx_height_m=10.0,
    )
    assert "fresnel_radius_1st_m" in res
    assert res["fresnel_radius_1st_m"] > 0
    assert "Clear" in res["status"]


def test_calculate_wall_attenuation():
    res = calculate_wall_attenuation(wall_type="concrete", frequency_band="5GHz", quantity=2)
    assert res["total_attenuation_db"] == 36.0
    assert res["wall_type"] == "concrete"


def test_optimize_ap_placement():
    res = optimize_ap_placement(
        length_m=40.0,
        width_m=20.0,
        wall_type="drywall",
        frequency_band="5GHz",
        min_rssi_dbm=-65.0,
    )
    assert res["recommended_ap_count"] >= 1
    assert len(res["ap_coordinates"]) == res["recommended_ap_count"]
    assert "suggested_channel_plan" in res


def test_calculate_link_budget_and_snr():
    res = calculate_link_budget_and_snr(rssi_dbm=-60.0, noise_floor_dbm=-95.0)
    assert res["snr_db"] == 35.0
    assert "MCS" in res["estimated_mcs_capability"]


def test_fetch_terrain_elevation_profile():
    from app.tools import fetch_terrain_elevation_profile

    res = fetch_terrain_elevation_profile(
        start_lat=34.0522,
        start_lon=-118.2437,
        end_lat=34.0600,
        end_lon=-118.2300,
        num_samples=5,
    )
    assert res["num_samples"] == 5
    assert len(res["elevation_profile"]) == 5
    assert "max_elevation_m" in res


def test_generate_terrain_rf_image():
    from unittest.mock import MagicMock
    from app.tools import generate_terrain_rf_image

    mock_context = MagicMock()
    res = generate_terrain_rf_image(
        prompt_description="A 2D layout of a transmitter tower on a hill.",
        tool_context=mock_context,
    )
    assert "status" in res or "error" in res
    if res.get("status") == "success":
        assert "https://storage.googleapis.com/terraincomm-assets-042a75196ccefd/generated_images/" in res["public_url"]
        assert mock_context.save_artifact.called


