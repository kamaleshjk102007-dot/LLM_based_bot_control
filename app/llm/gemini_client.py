from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai import errors, types

from app.commands.models import Action, UniversalCommand
from app.config.settings import Settings


class GeminiCommandError(RuntimeError):
    """Raised when Gemini cannot produce a valid UniversalCommand."""


def gemini_response_json_schema() -> dict[str, Any]:
    """Return the simplified JSON schema used for Gemini structured output."""

    return {
        "type": "object",
        "properties": {
            "version": {
                "type": "string",
                "enum": ["1.0"],
            },
            "robot_id": {
                "type": "string",
                "nullable": True,
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "MOVE",
                                "ROTATE",
                                "STOP",
                                "HOME",
                                "PICK",
                                "PLACE",
                                "GRIP",
                                "RELEASE",
                                "NAVIGATE",
                                "GET_STATUS",
                            ],
                        },
                        "object": {
                            "type": "object",
                            "nullable": True,
                        },
                        "target": {
                            "type": "object",
                            "nullable": True,
                        },
                        "position": {
                            "type": "string",
                            "nullable": True,
                        },
                        "direction": {
                            "type": "string",
                            "nullable": True,
                        },
                        "distance": {
                            "type": "number",
                            "nullable": True,
                        },
                        "angle": {
                            "type": "number",
                            "nullable": True,
                        },
                        "unit": {
                            "type": "string",
                            "nullable": True,
                        },
                        "parameters": {
                            "type": "object",
                            "nullable": True,
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["version", "tasks"],
    }


MAX_RETRIES = 5
RETRY_DELAYS = (2, 4, 8, 16, 30)


class GeminiClient:
    """Gemini natural-language → UniversalCommand client."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:

        self.settings = settings

        if not settings.gemini_api_key:
            raise GeminiCommandError(
                "Gemini API key is missing. "
                "Set GEMINI_API_KEY in the environment."
            )

        # Allows the test suite to inject a mocked Gemini client.
        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(
                api_key=settings.gemini_api_key
            )

        self.model = settings.gemini_model

    # ================================================================
    # ERROR HANDLING
    # ================================================================

    def _safe_client_error(
        self,
        exc: Exception,
    ) -> GeminiCommandError:

        message = str(exc)

        if "401" in message or "API key" in message:
            return GeminiCommandError(
                "Gemini authentication failed. "
                "Check GEMINI_API_KEY."
            )

        if "403" in message:
            return GeminiCommandError(
                "Gemini access was denied. "
                "Check API permissions and model access."
            )

        if "429" in message:
            return GeminiCommandError(
                "Gemini rate limit reached. "
                "Please try again later."
            )

        if "404" in message:
            return GeminiCommandError(
                f"Gemini model '{self.model}' was not found."
            )

        return GeminiCommandError(
            f"Gemini API request failed: {message}"
        )

    # ================================================================
    # PROMPT
    # ================================================================

    def _build_prompt(
        self,
        user_text: str,
        perception_context: str | None = None,
    ) -> str:

        perception_section = ""
        if perception_context:
            perception_section = f"""
PERCEPTION / DETECTED OBJECTS ON TABLE:
{perception_context.strip()}

PICK and PLACE RULES:
- When the user asks to pick or grasp an object (e.g., "pick up the red cube"), generate a PICK task:
  - Set "object" with matching descriptors (e.g., color, type).
  - Include the detected coordinates in "parameters": {{"x": <X>, "y": <Y>, "z": <Z>}}.
- When the user asks to place or put an object at a target (e.g., "place it in the green bin"), generate a PLACE task:
  - Set "target" with matching descriptors (e.g., color, type).
  - Include the target coordinates in "parameters": {{"x": <X>, "y": <Y>, "z": <Z>}}.
- When the user asks to move an object to a destination (e.g., "move the red block to the green bin"):
  - Generate a 2-task sequence:
    1. PICK the source object with its coordinates in parameters.
    2. PLACE at the target destination with its coordinates in parameters.
"""

        return f"""
You are a robot command parser.

Convert the user's natural-language instruction into
exactly one JSON object matching the UniversalCommand schema.

IMPORTANT RULES:

1. Do not execute the robot.
2. Return ONLY JSON.
3. Use version "1.0".

Available actions:

MOVE
ROTATE
STOP
HOME
PICK
PLACE
GRIP
RELEASE
NAVIGATE
GET_STATUS
{perception_section}
MOVE:

- Cartesian axes are X, Y and Z.
- Direction can be X, Y, Z, +X, -X, +Y, -Y, +Z, -Z.
- Distance should normally be positive.
- Preserve the axis specified by the user.
- Preserve the sign/direction specified by the user.
- If the user says "MOVE -5 mm on Y",
  represent it as direction "-Y" and distance 5.
- If the user says "MOVE +5 mm on Y",
  represent it as direction "Y" and distance 5.
- If the user says "MOVE -5 mm on X",
  represent it as direction "-X" and distance 5.
- If the user says "MOVE +5 mm on X",
  represent it as direction "X" and distance 5.
- If the user says "MOVE -5 mm on Z",
  represent it as direction "-Z" and distance 5.
- If the user specifies absolute coordinates (e.g. "Move to X 220, Y -260, Z 25" or "Go to X=220, Y=-260, Z=25"):
  represent it as:
  {{
      "action": "MOVE",
      "position": "coordinate_target",
      "parameters": {{"x": 220.0, "y": -260.0, "z": 25.0}}
  }}

ROTATE:

- Use R, +R or -R.
- Angle should normally be positive.
- Negative rotation should be represented by -R.

Examples:

{{
    "version": "1.0",
    "robot_id": null,
    "tasks": [
        {{
            "action": "MOVE",
            "direction": "X",
            "distance": 5,
            "unit": "mm"
        }}
    ]
}}

For:

MOVE -5 mm on Y

the expected task is:

{{
    "action": "MOVE",
    "direction": "-Y",
    "distance": 5,
    "unit": "mm"
}}

For:

MOVE +5 mm on Y

the expected task is:

{{
    "action": "MOVE",
    "direction": "Y",
    "distance": 5,
    "unit": "mm"
}}

User instruction:
{user_text.strip()}
"""

    # ================================================================
    # RESPONSE EXTRACTION
    # ================================================================

    @staticmethod
    def _extract_response_data(
        response: Any,
    ) -> Any:

        parsed = getattr(
            response,
            "parsed",
            None,
        )

        if parsed is not None:
            return parsed

        text = getattr(
            response,
            "text",
            None,
        )

        if text is None or not str(text).strip():
            raise GeminiCommandError(
                "Gemini returned an empty response."
            )

        try:
            return json.loads(
                str(text).strip()
            )

        except json.JSONDecodeError as exc:
            raise GeminiCommandError(
                "Gemini returned invalid JSON."
            ) from exc

    # ================================================================
    # EXTRACT AXES FROM USER INSTRUCTION
    # ================================================================

    @staticmethod
    def _extract_axes(
        user_text: str,
    ) -> list[str]:

        text = user_text.upper()

        axes: list[str] = []

        pattern = re.compile(
            r"""
            [+-]\s*([XYZ])\b
            |
            \bon\s+(?:the\s+)?([XYZ])\b
            |
            \baxis\s*([XYZ])\b
            |
            \b([XYZ])\s*axis\b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for match in pattern.finditer(text):

            axis = next(
                (
                    value
                    for value in match.groups()
                    if value is not None
                ),
                None,
            )

            if axis:
                axes.append(
                    axis.upper()
                )

        return axes

    # ================================================================
    # EXTRACT SIGNED AXES
    # ================================================================

    @staticmethod
    def _extract_signed_axes(
        user_text: str,
    ) -> list[tuple[str, int]]:

        text = user_text.upper()

        result: list[tuple[str, int]] = []

        pattern = re.compile(
            r"([+-])\s*([XYZ])\b"
        )

        for match in pattern.finditer(text):

            sign = (
                1
                if match.group(1) == "+"
                else -1
            )

            axis = match.group(2).upper()

            result.append(
                (axis, sign)
            )

        return result

    # ================================================================
    # EXTRACT SIGNED DISTANCE + AXIS
    # ================================================================

    @staticmethod
    def _extract_signed_distance_axes(
        user_text: str,
    ) -> list[tuple[str, int]]:

        """
        Detect commands such as:

            MOVE -5 mm on Y
            MOVE +5 mm on Y
            MOVE -10 mm on X
            MOVE +2.5 mm on Z

        Returns:

            [
                ("Y", -1)
            ]

        The distance itself is NOT returned here.
        Only the sign and axis are extracted.
        """

        text = user_text.upper()

        result: list[tuple[str, int]] = []

        pattern = re.compile(
            r"""
            ([+-])
            \s*
            (?:\d+(?:\.\d*)?|\.\d+)
            \s*
            (?:MM|MILLIMETER|MILLIMETERS|
               CM|CENTIMETER|CENTIMETERS|
               M|METER|METERS)?
            \s*
            (?:ON|ALONG|IN)
            \s*
            (?:THE\s*)?
            ([XYZ])\b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for match in pattern.finditer(text):

            sign = (
                1
                if match.group(1) == "+"
                else -1
            )

            axis = match.group(2).upper()

            result.append(
                (axis, sign)
            )

        return result

    # ================================================================
    # PROVIDER RESPONSE REPAIR
    # ================================================================

    def _repair_provider_data(
        self,
        data: Any,
        user_text: str,
    ) -> dict[str, Any]:

        if not isinstance(data, dict):

            raise GeminiCommandError(
                "Gemini returned an unsupported response format."
            )

        repaired = dict(data)

        raw_tasks = repaired.get(
            "tasks"
        )

        if not isinstance(
            raw_tasks,
            list,
        ):

            raise GeminiCommandError(
                "Gemini response must contain tasks."
            )

        if not raw_tasks:

            raise GeminiCommandError(
                "Gemini response must contain at least one task."
            )

        tasks: list[dict[str, Any]] = []

        explicit_axes = self._extract_axes(
            user_text
        )

        signed_axes = self._extract_signed_axes(
            user_text
        )

        signed_distance_axes = (
            self._extract_signed_distance_axes(
                user_text
            )
        )

        for index, raw_task in enumerate(
            raw_tasks
        ):

            if not isinstance(
                raw_task,
                dict,
            ):

                raise GeminiCommandError(
                    "Gemini returned an invalid task."
                )

            task = dict(
                raw_task
            )

            action = str(
                task.get(
                    "action",
                    "",
                )
            ).strip().upper()

            task["action"] = action

            # ========================================================
            # ROTATE → MOVE MISCLASSIFICATION
            # ========================================================

            unit = task.get(
                "unit"
            )

            linear_units = {
                "mm",
                "millimeter",
                "millimeters",
                "cm",
                "centimeter",
                "centimeters",
                "m",
                "meter",
                "meters",
            }

            if (
                action == "ROTATE"
                and task.get(
                    "distance"
                ) is not None
                and unit is not None
                and str(unit).lower()
                in linear_units
            ):

                task["action"] = "MOVE"

                action = "MOVE"

                if task.get(
                    "direction"
                ) is None:

                    lowered = user_text.lower()

                    if "downward" in lowered:
                        task["direction"] = "downward"

                    elif "upward" in lowered:
                        task["direction"] = "upward"

                    elif "forward" in lowered:
                        task["direction"] = "forward"

                    elif "backward" in lowered:
                        task["direction"] = "backward"

            # ========================================================
            # MOVE
            # ========================================================

            if action == "MOVE":

                distance = task.get(
                    "distance"
                )

                direction = task.get(
                    "direction"
                )

                # ----------------------------------------------------
                # Remember whether provider gave negative distance.
                # ----------------------------------------------------

                negative_distance = False

                if distance is not None:

                    try:

                        distance_value = float(
                            distance
                        )

                    except (
                        TypeError,
                        ValueError,
                    ) as exc:

                        raise GeminiCommandError(
                            "MOVE distance must be numeric."
                        ) from exc

                    if distance_value < 0:

                        negative_distance = True

                        distance_value = abs(
                            distance_value
                        )

                        # --------------------------------------------
                        # Negative distance + provider direction
                        # --------------------------------------------

                        if direction is not None:

                            direction_text = str(
                                direction
                            ).strip().upper()

                            if direction_text.startswith(
                                "+"
                            ):

                                direction_text = (
                                    "-"
                                    + direction_text[1:]
                                )

                            elif direction_text.startswith(
                                "-"
                            ):

                                direction_text = (
                                    "+"
                                    + direction_text[1:]
                                )

                            else:

                                direction_text = (
                                    "-"
                                    + direction_text
                                )

                            task["direction"] = (
                                direction_text
                            )

                        # --------------------------------------------
                        # Negative distance + no provider direction
                        # --------------------------------------------

                        elif index < len(
                            signed_axes
                        ):

                            axis, sign = (
                                signed_axes[index]
                            )

                            task["direction"] = (
                                axis
                                if sign > 0
                                else "-" + axis
                            )

                        task["distance"] = (
                            distance_value
                        )

                    else:

                        task["distance"] = (
                            distance_value
                        )

                # ----------------------------------------------------
                # Explicit axis restoration
                # ----------------------------------------------------

                if index < len(
                    explicit_axes
                ):

                    axis = explicit_axes[index]

                    # =================================================
                    # IMPORTANT FIX:
                    #
                    # Handle:
                    #
                    # MOVE -5 mm on Y
                    # MOVE +5 mm on Y
                    #
                    # The sign belongs to the DISTANCE, not directly
                    # to the axis.
                    # =================================================

                    distance_sign = None

                    if index < len(
                        signed_distance_axes
                    ):

                        signed_distance_axis, signed_distance_sign = (
                            signed_distance_axes[index]
                        )

                        if signed_distance_axis == axis:

                            distance_sign = (
                                signed_distance_sign
                            )

                    # -------------------------------------------------
                    # If user explicitly said:
                    #
                    # MOVE -5 mm on Y
                    #
                    # force direction to -Y.
                    # -------------------------------------------------

                    if distance_sign == -1:

                        task["direction"] = (
                            "-" + axis
                        )

                    # -------------------------------------------------
                    # If user explicitly said:
                    #
                    # MOVE +5 mm on Y
                    #
                    # force direction to Y.
                    # -------------------------------------------------

                    elif distance_sign == 1:

                        task["direction"] = (
                            axis
                        )

                    # -------------------------------------------------
                    # If provider supplied a negative distance,
                    # preserve that converted direction.
                    # -------------------------------------------------

                    elif negative_distance:

                        existing_direction = task.get(
                            "direction"
                        )

                        if existing_direction is not None:

                            direction_text = str(
                                existing_direction
                            ).strip().upper()

                            if direction_text.startswith(
                                "-"
                            ):

                                task["direction"] = (
                                    "-" + axis
                                )

                            else:

                                task["direction"] = (
                                    axis
                                )

                        else:

                            task["direction"] = (
                                "-" + axis
                            )

                    else:

                        # ------------------------------------------------
                        # Check if the USER explicitly specified
                        # a signed axis such as -X.
                        # ------------------------------------------------

                        sign = None

                        if index < len(
                            signed_axes
                        ):

                            signed_axis, signed_sign = (
                                signed_axes[index]
                            )

                            if signed_axis == axis:

                                sign = signed_sign

                        if sign == -1:

                            task["direction"] = (
                                "-" + axis
                            )

                        else:

                            # Existing project convention:
                            #
                            # +X is represented as X
                            # +Y is represented as Y
                            # +Z is represented as Z

                            task["direction"] = (
                                axis
                            )

                # ----------------------------------------------------
                # No explicit axis in user instruction.
                # Preserve provider direction.
                # ----------------------------------------------------

                elif direction is not None:

                    direction_text = str(
                        direction
                    ).strip().upper()

                    if direction_text in {
                        "X",
                        "Y",
                        "Z",
                    }:

                        task["direction"] = (
                            direction_text
                        )

                    elif direction_text in {
                        "+X",
                        "-X",
                        "+Y",
                        "-Y",
                        "+Z",
                        "-Z",
                    }:

                        task["direction"] = (
                            direction_text
                        )

            # ========================================================
            # ROTATE
            # ========================================================

            elif action == "ROTATE":

                angle = task.get(
                    "angle"
                )

                direction = task.get(
                    "direction"
                )

                if angle is not None:

                    try:

                        angle_value = float(
                            angle
                        )

                    except (
                        TypeError,
                        ValueError,
                    ) as exc:

                        raise GeminiCommandError(
                            "ROTATE angle must be numeric."
                        ) from exc

                    if angle_value < 0:

                        angle_value = abs(
                            angle_value
                        )

                        direction_text = str(
                            direction or "R"
                        ).strip().upper()

                        if direction_text in {
                            "R",
                            "+R",
                        }:

                            direction_text = "-R"

                        elif direction_text == "-R":

                            direction_text = "+R"

                        task["direction"] = (
                            direction_text
                        )

                        task["angle"] = (
                            angle_value
                        )

                    else:

                        task["angle"] = (
                            angle_value
                        )

                        if direction is not None:

                            direction_text = str(
                                direction
                            ).strip().upper()

                            if direction_text in {
                                "R",
                                "+R",
                                "-R",
                            }:

                                task["direction"] = (
                                    direction_text
                                )

            if action == "PICK":
                allowed_pick = {"action", "object", "parameters"}
                task = {k: v for k, v in task.items() if k in allowed_pick and v is not None}
                obj = task.get("object")
                if isinstance(obj, dict) and not any(v for v in obj.values() if v is not None):
                    task["object"] = {"name": "target_object"}
            elif action == "PLACE":
                allowed_place = {"action", "target", "parameters"}
                task = {k: v for k, v in task.items() if k in allowed_place and v is not None}
                tgt = task.get("target")
                if isinstance(tgt, dict) and not any(v for v in tgt.values() if v is not None):
                    task["target"] = {"name": "target_destination"}

            tasks.append(
                task
            )

        repaired["tasks"] = tasks

        return repaired

    # ================================================================
    # MAIN GEMINI METHOD
    # ================================================================

    def generate_command(
        self,
        user_text: str,
        perception_context: str | None = None,
    ) -> UniversalCommand:

        if not user_text or not user_text.strip():

            raise GeminiCommandError(
                "Robot instruction cannot be empty."
            )

        prompt = self._build_prompt(
            user_text,
            perception_context=perception_context,
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_response_json_schema(),
        )

        response: Any = None

        # ============================================================
        # GEMINI API WITH 503 RETRY
        # ============================================================

        for attempt in range(
            MAX_RETRIES
        ):

            try:

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                )

                break

            except errors.ClientError as exc:

                raise self._safe_client_error(
                    exc
                ) from exc

            except Exception as exc:

                error_code = getattr(
                    exc,
                    "code",
                    None,
                )

                error_type = type(
                    exc
                ).__name__

                temporary = (
                    error_code == 503
                    or error_type == "ServerError"
                    or "503" in str(exc)
                    or "UNAVAILABLE" in str(exc)
                )

                if temporary:

                    if attempt < MAX_RETRIES - 1:

                        delay = RETRY_DELAYS[
                            attempt
                        ]

                        print(
                            f"\nGemini temporarily unavailable "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})."
                        )

                        print(
                            f"Retrying in {delay} seconds..."
                        )

                        time.sleep(
                            delay
                        )

                        continue

                    raise GeminiCommandError(
                        "Gemini is temporarily unavailable "
                        "after multiple retry attempts. "
                        "Please try again later."
                    ) from exc

                raise GeminiCommandError(
                    "Gemini API request failed: "
                    f"{error_type}: {exc}"
                ) from exc

        if response is None:

            raise GeminiCommandError(
                "Gemini did not return a response."
            )

        # ============================================================
        # EXTRACT GEMINI RESPONSE
        # ============================================================

        provider_data = (
            self._extract_response_data(
                response
            )
        )

        # ============================================================
        # REPAIR BEFORE VALIDATION
        # ============================================================

        repaired_data = (
            self._repair_provider_data(
                provider_data,
                user_text,
            )
        )

        # ============================================================
        # FINAL PYDANTIC VALIDATION
        # ============================================================

        try:

            command = (
                UniversalCommand.model_validate(
                    repaired_data
                )
            )

        except Exception as exc:

            raise GeminiCommandError(
                "Gemini returned JSON that does not match "
                f"the UniversalCommand schema: {exc}"
            ) from exc

        return command


# Existing project/test compatibility
GeminiCommandClient = GeminiClient