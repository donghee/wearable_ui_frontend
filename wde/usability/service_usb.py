from io import BytesIO
from flask import jsonify, request, send_file
from flask_restx import Namespace, Resource
import sys
import os
import base64


from .extend.usability.run_usability_system import draw_usability_graph, run_usability_system
usability_ns = Namespace("usability", path="/api/usability", description="Usability service")


@usability_ns.route("/score")
class UsabilityScore(Resource):
    def get(self):
        # 예시: 쿼리 파라미터 사용 가능
        user_id = request.args.get("user_id", None)
        # 실제 데이터 로직은 필요에 따라 구현
        timeout = 10
        error_rate = 5
        force = int(error_rate)
        noise = 5
        task = "adaptive"
        age = 30
        total_score = run_usability_system(timeout, [task, "adaptive"], age, force, noise)
        result = {"score": total_score[0], "user_id": user_id}
        return result


@usability_ns.route("/graph/<string:task>/<int:age>/<int:error_rate>")
class UsabilityGraph(Resource):
    def get(self, task, age, error_rate):
        timeout = 10
        force = int(error_rate)
        noise = 5
        img = draw_usability_graph(timeout, [task, "adaptive"], age, force, noise)

        return send_file(img, mimetype='image/png')
