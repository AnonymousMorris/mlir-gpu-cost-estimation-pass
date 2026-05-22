from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .equation import build_cost_equation, evaluate_equation
from .mlir_parser import CostParseError, parse_cost_function


class ParseRequest(BaseModel):
    mlir: str


class EvaluateRequest(BaseModel):
    mlir: str
    values: dict[str, float]


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MLIR Cost UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.post("/api/parse")
def parse_cost(request: ParseRequest) -> dict:
    try:
        equation = build_cost_equation(parse_cost_function(request.mlir))
    except CostParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "variables": equation.variables,
        "equationText": equation.equation_text,
        "javascriptParameters": equation.javascript_parameters,
        "javascriptExpression": equation.javascript_expression,
    }


@app.post("/api/evaluate")
def evaluate_cost(request: EvaluateRequest) -> dict:
    try:
        equation = build_cost_equation(parse_cost_function(request.mlir))
        cost = evaluate_equation(equation, request.values)
    except (CostParseError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"cost": cost}
