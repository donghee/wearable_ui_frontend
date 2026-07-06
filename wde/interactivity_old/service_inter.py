from flask import jsonify, request, send_file
from flask_restx import Namespace, Resource
import os
import pandas as pd
import sys
import os

from .extend.GET_interaction_score_exhibition import combined_torques_graph, compare_value, map_motion_percent
import base64
from io import BytesIO

interactivity_ns = Namespace("interactivity", path="/api/interactivity", description="Interactivity service")


@interactivity_ns.route("/score")
class InteractivityScore(Resource):
    def get(self):
        # 예시: 쿼리 파라미터 사용 가능
        user_id = request.args.get("user_id", None)
        # 실제 데이터 로직은 필요에 따라 구현
        result = {"score": 95, "user_id": user_id}
        return result


@interactivity_ns.route("/graph")
class InteractivityGraph(Resource):
    def get(self):
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extend", "interaction_control_node_1.csv")
        df = pd.read_csv(file_path)
        df = map_motion_percent(df)
        df = df.sort_values('Motion')

        evaluate_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extend", "Evaluation_Table2.csv")
        df_evaluate = pd.read_csv(evaluate_file_path)
        total_score = compare_value(df)
        img = combined_torques_graph(df, df_evaluate)

        # 이미지 객체를 base64로 변환
        if hasattr(img, "save"):
            buf = BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()
        else:
            # img is already BytesIO
            img_bytes = img.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_url = f"data:image/png;base64,{img_base64}"

        return {"total_score": total_score, "image": img_url}
