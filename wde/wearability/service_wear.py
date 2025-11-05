from flask import jsonify, request, send_file
from flask_restx import Namespace, Resource
from io import BytesIO

import base64

from .extend.device_1DOF.python.wearability.device_1DOF_timegraph import timegraph
from .extend.device_1DOF.python.wearability.device_1DOF_totalgraph import totalgraph

wearability_ns = Namespace("wearability", path="/api/wearability", description="Wearability service")


@wearability_ns.route("/graph/total")
class WearabilityTotalGraph(Resource):
    def get(self):
        img = totalgraph()
        return send_file(img, mimetype='image/png')


@wearability_ns.route("/graph/time/<int:wear_case>/<int:line>")
class WearabilityTimeGraph(Resource):
    def get(self, wear_case, line):
        img = timegraph(int(wear_case), int(line))

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

        return jsonify({"image": img_url})
