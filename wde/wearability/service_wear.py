from flask import jsonify, request, send_file
from flask_restx import Namespace, Resource
from io import BytesIO
import base64
import os
import xml.etree.ElementTree as ET

from .extend.device_1DOF.python.wearability.device_1DOF_timegraph import timegraph
from .extend.device_1DOF.python.wearability.device_1DOF_totalgraph import totalgraph, totalscore

wearability_ns = Namespace("wearability", path="/api/wearability", description="Wearability service")

index_dat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Index.dat")
print (f"Index.dat path: {index_dat}")


def load_model(xml_path):
    root = ET.parse(xml_path).getroot()
    # patient model
    patient_body_ = root.find("worldbody").find("body").find("body").find("body[@name='r_humerus']")
    user_attrib = patient_body_.attrib.get('user').split(' ')
    patient_user_attributes = {
        'L_h_u': float(user_attrib[11]),
        'm_h_u': float(user_attrib[12]),
        'G_h_u': float(user_attrib[13]),
        'L_h_f': float(user_attrib[14]),
        'm_h_f': float(user_attrib[15]),
        'G_h_f': float(user_attrib[16]),
    }

    # exoskeleton device model
    exo_body_ = root.find("worldbody").find("body[@name='exo_anchor_upper']")
    user_attrib = exo_body_.attrib.get('user').split(' ')
    exo_user_attributes = {
        'K_u': float(user_attrib[0]),
        'K_f': float(user_attrib[0]),
        'K_t_u': float(user_attrib[2]),
        'K_t_f': float(user_attrib[2]),
        'L_d_u': float(user_attrib[3]),
        'm_d_u': float(user_attrib[4]),
        'G_d_u': float(user_attrib[5]),
        'L_d_f': float(user_attrib[6]),
        'm_d_f': float(user_attrib[7]),
        'G_d_f': float(user_attrib[8]),
    }
    return {**patient_user_attributes, **exo_user_attributes}

def read_index_dat(file_path):
    selected_patient = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        selected_patient['name'] = lines[0].strip()
        selected_patient['ready_to_xml'] = lines[1].strip() # 1 is ready, 2 is not ready
        selected_patient['ready_to_result.csv'] = lines[2].strip() # 1 is completed, 2 is not completed

    return selected_patient

@wearability_ns.route("/patient")
class WearabilityGetPatientInfo(Resource):
    def get(self):
        selected_patient = read_index_dat(index_dat)
        name = selected_patient['name']
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Patient/", name, name + ".xml")
        print(f"Loading model from: {xml_path}")
        user_attributes = load_model(xml_path)
        selected_patient.update(user_attributes)
        return jsonify(selected_patient)

#  @wearability_ns.route("/graph/total")
#  class WearabilityTotalGraph(Resource):
#      def get(self):
#          selected_patient = read_index_dat(index_dat)
#          name = selected_patient['name']
#          xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Patient/", name, name + ".xml")
#          print(f"Loading model from: {xml_path}")
#          input_parameter = load_model(xml_path)
#          img = totalgraph(input_parameter)
#          return send_file(img, mimetype='image/png')

# TODO fix url to use /graph/total instead of graph/time
@wearability_ns.route("/graph/time/<int:wear_case>/<int:line>")
class WearabilityTotalGraph(Resource):
    def get(self, wear_case, line):
        selected_patient = read_index_dat(index_dat)
        name = selected_patient['name']
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Patient/", name, name + ".xml")
        print(f"Loading model from: {xml_path}")
        input_parameter = load_model(xml_path)
        img = totalgraph(input_parameter)
        return send_file(img, mimetype='image/png')

@wearability_ns.route("/score")
class WearabilityScore(Resource):
    def get(self):
        user_id = request.args.get("user_id", None)
        selected_patient = read_index_dat(index_dat)
        name = selected_patient['name']
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Patient/", name, name + ".xml")
        input_parameter = load_model(xml_path)
        total_score = totalscore(input_parameter)
        result = {"score": total_score * 100, "user_id": user_id}
        return result

#  @wearability_ns.route("/graph/time/<int:wear_case>/<int:line>")
#  class WearabilityTimeGraph(Resource):
#      def get(self, wear_case, line):
#          img = timegraph(int(wear_case), int(line))
#
#          return send_file(img, mimetype='image/png')

#  @wearability_ns.route("/graph/time2/<int:wear_case>/<int:line>")
#  class WearabilityTimeGraph2(Resource):
#      def get(self, wear_case, line):
#          img = timegraph(int(wear_case), int(line))
#          return send_file(img, mimetype='image/png')
