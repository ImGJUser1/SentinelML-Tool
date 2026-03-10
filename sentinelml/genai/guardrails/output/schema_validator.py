import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/output/schema_validator.py
"""
Structured output validation using Pydantic schemas.
"""

import json
from typing import Any, Dict, List, Optional, Type

from sentinelml.core.base import BaseGuardrail


class SchemaValidator(BaseGuardrail):
    """
    Validate LLM outputs against structured schemas.

    Ensures JSON outputs conform to expected structure,
    types, and constraints using Pydantic models.

    Parameters
    ----------
    schema : Type[BaseModel] or dict
        Pydantic model class or JSON schema dict.
        repair : bool, default=False
        Attempt to repair invalid outputs.
    max_repair_attempts : int, default=3
        Maximum repair iterations.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class Product(BaseModel):
    ...     name: str
    ...     price: float
    ...     in_stock: bool

    >>> validator = SchemaValidator(schema=Product)
    >>> result = validator.validate('{"name": "Widget", "price": 9.99}')
    """

    def __init__(
        self,
        schema: Optional[Type] = None,
        json_schema: Optional[Dict] = None,
        name: str = "SchemaValidator",
        fail_mode: str = "filter",
        repair: bool = False,
        max_repair_attempts: int = 3,
        repair_model: Optional[Callable] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)

        if schema is not None:
            self.schema = schema
            self.schema_type = "pydantic"
        elif json_schema is not None:
            self.schema = json_schema
            self.schema_type = "json"
        else:
            raise ValueError("Either schema or json_schema must be provided")

        self.repair = repair
        self.max_repair_attempts = max_repair_attempts
        self.repair_model = repair_model

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate content against schema.

        Returns
        -------
        dict with validation results.
        """
        # Try to parse as JSON
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
        except json.JSONDecodeError as e:
            if self.repair and self.repair_model:
                return self._attempt_repair(content, str(e))
            return {
                "is_valid": False,
                "score": 0.0,
                "action": self.fail_mode,
                "metadata": {"error": "Invalid JSON", "details": str(e)},
            }

        # Validate against schema
        if self.schema_type == "pydantic":
            return self._validate_pydantic(data)
        else:
            return self._validate_json_schema(data)

    def _validate_pydantic(self, data: Dict) -> Dict[str, Any]:
        """Validate using Pydantic model."""
        try:
            validated = self.schema(**data)
            return {
                "is_valid": True,
                "score": 1.0,
                "action": "pass",
                "metadata": {
                    "validated_data": validated.dict()
                    if hasattr(validated, "dict")
                    else validated.model_dump(),
                    "schema": self.schema.__name__,
                },
            }
        except Exception as e:
            errors = self._extract_pydantic_errors(e)
            if self.repair and self.repair_model:
                return self._attempt_repair(json.dumps(data), str(e))

            return {
                "is_valid": False,
                "score": 0.0,
                "action": self.fail_mode,
                "metadata": {"error": "Schema validation failed", "validation_errors": errors},
            }

    def _validate_json_schema(self, data: Dict) -> Dict[str, Any]:
        """Validate using JSON schema."""
        try:
            from jsonschema import ValidationError, validate

            validate(instance=data, schema=self.schema)
            return {
                "is_valid": True,
                "score": 1.0,
                "action": "pass",
                "metadata": {"schema_valid": True},
            }
        except ValidationError as e:
            if self.repair and self.repair_model:
                return self._attempt_repair(json.dumps(data), str(e))

            return {
                "is_valid": False,
                "score": 0.0,
                "action": self.fail_mode,
                "metadata": {
                    "error": "JSON schema validation failed",
                    "details": e.message,
                    "path": list(e.path),
                },
            }
        except ImportError:
            raise ImportError("jsonschema required for JSON schema validation")

    def _attempt_repair(self, content: str, error_msg: str) -> Dict[str, Any]:
        """Attempt to repair invalid output."""
        for attempt in range(self.max_repair_attempts):
            try:
                repair_prompt = f"""
                The following JSON is invalid: {error_msg}

                Original content:
                {content}

                Please fix the JSON and return only the corrected JSON.
                """

                repaired = self.repair_model(repair_prompt)

                # Try to validate repaired content
                result = self.validate(repaired)
                if result["is_valid"]:
                    result["metadata"]["repaired"] = True
                    result["metadata"]["repair_attempts"] = attempt + 1
                    return result

                content = repaired

            except Exception as e:
                continue

        return {
            "is_valid": False,
            "score": 0.0,
            "action": self.fail_mode,
            "metadata": {
                "error": "Repair failed",
                "original_error": error_msg,
                "attempts": self.max_repair_attempts,
            },
        }

    def _extract_pydantic_errors(self, error) -> List[Dict]:
        """Extract structured error information from Pydantic."""
        errors = []
        if hasattr(error, "errors"):
            for err in error.errors():
                errors.append(
                    {
                        "loc": err.get("loc", []),
                        "msg": err.get("msg", ""),
                        "type": err.get("type", ""),
                    }
                )
        else:
            errors.append({"msg": str(error)})
        return errors
