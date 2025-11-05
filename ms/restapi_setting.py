from flask import request
from flask_restx import Namespace, Resource, fields
from sqlite3 import IntegrityError

from .db import (
	_row_to_dict,
	create_human,
	list_human,
	read_human,
	update_human,
	delete_human,
	create_device,
	list_device,
	read_device,
	update_device,
	delete_device,
)

setting_ns = Namespace("setting", path="/api/setting", description="Settings API for humans and devices")

human_fields = setting_ns.model('Human', {
    'no': fields.Integer(required=True, description='no'),
    'name': fields.String(required=True, description='name'),
    'tags': fields.String(description='tags'),
    'age': fields.String(description='나이'),
    'gender': fields.String(description='성별'),
    'tendency': fields.String(description='성향'),
    'alpha': fields.String(description='alpha'),
    'beta': fields.String(description='beta'),
    'gamma': fields.String(description='gamma'),
    'n': fields.String(description='n'),
    'rgain': fields.String(description='r-gain'),
    'fmax': fields.String(description='f-max'),
    'kpass': fields.String(description='k-pass'),
    'lopt': fields.String(description='l-opt'),
    'upperarm_length': fields.String(description='상완 길이'),
    'upperarm_mass': fields.String(description='상완 질량'),
    'upperarm_location': fields.String(description='상완 무게중심 위치'),
    'forearm_length': fields.String(description='전완 길이'),
    'forearm_mass': fields.String(description='전완 질량'),
    'forearm_location': fields.String(description='전완 무게중심 위치'),
    'arm_circumference': fields.String(description='팔 둘레'),
    'leg_circumference': fields.String(description='다리 둘레'),
    'chest_circumference': fields.String(description='가슴 둘레'),
    'belly_circumference': fields.String(description='배 둘레'),
    'body_length': fields.String(description='몸 길이'),
    'thigh_length': fields.String(description='허벅지 길이'),
    'calf_length': fields.String(description='종아리 길이'),
    'is_default': fields.String(description='기본 데이터 여부'),
    'reg_date': fields.String(description='등록 날짜'),
})

device_fields = setting_ns.model('Device', {
    'no': fields.Integer(required=True, description='no'),
    'name': fields.String(required=True, description='name'),
    'tags': fields.String(description='tags'),
    'motor_mass': fields.String(description='모터 질량'),
    'motor_angle_upperlimit': fields.String(description='모터 상한 각도'),
    'motor_angle_lowerlimit': fields.String(description='모터 하한 각도'),
    'motor_damping': fields.String(description='모터 감쇠'),
    'motor_friction': fields.String(description='모터 마찰'),
    'velcro_shear_stiffness': fields.String(description='전단방향 강성'),
    'velcro_vertical_stiffness': fields.String(description='수직 방향 강성'),
    'velcro_torsional_stiffness': fields.String(description='비틀림 강성'),
    'velcro_strength_steel': fields.String(description='착용강도'),
    'velcro_wear_damping': fields.String(description='착용댐핑'),
    'velcro_wearing_space': fields.String(description='착용유격'),
    'upperarm_length': fields.String(description='상완 길이'),
    'upperarm_mass': fields.String(description='상완 질량'),
    'upperarm_location': fields.String(description='상완 무게중심 위치'),
    'forearm_length': fields.String(description='전완 길이'),
    'forearm_mass': fields.String(description='전완 질량'),
    'forearm_location': fields.String(description='전완 무게중심 위치'),
    'shear_direction_wear_error': fields.String(description='전단방향 착용오차'),
    'vertical_direction_wear_error': fields.String(description='수직방향 착용오차'),
    'torsional_direction_wear_error': fields.String(description='비틀림방향 착용오차'),
    'is_default': fields.String(description='기본 데이터 여부'),
    'reg_date': fields.String(description='등록 날짜'),
})
    

# ---------------------- tbl_human ----------------------
@setting_ns.route("/humans")
class Humans(Resource):
	@setting_ns.marshal_list_with(human_fields)
	def get(self):
		rows = list_human()
		return [_row_to_dict("tbl_human", r) for r in rows]

	@setting_ns.expect(human_fields)
	def post(self):
		data = request.json or {}
		if "name" not in data:
			return {"error": "'name' is required"}, 400
		try:
			create_human(data)
			return {"success": True}, 201
		except IntegrityError as e:
			return {"error": f"Integrity error: {str(e)}"}, 400
		except Exception as e:
			return {"error": str(e)}, 500


@setting_ns.route("/humans/<int:no>")
class HumanItem(Resource):
	@setting_ns.marshal_with(human_fields)
	def get(self, no):
		row = read_human(no)
		if not row:
			return {"error": "Not found"}, 404
		return _row_to_dict("tbl_human", row)

	@setting_ns.expect(human_fields)
	def put(self, no):
		data = request.json or {}
		if not data:
			return {"error": "No fields to update"}, 400
		try:
			update_human(no, data)
			row = read_human(no)
			if not row:
				return {"error": "Not found"}, 404
			return _row_to_dict("tbl_human", row)
		except Exception as e:
			return {"error": str(e)}, 500

	def delete(self, no):
		try:
			row = read_human(no)
			if not row:
				return {"error": "Not found"}, 404
			delete_human(no)
			return {"success": True}
		except Exception as e:
			return {"error": str(e)}, 500


# ---------------------- tbl_device ----------------------
@setting_ns.route("/devices")
class Devices(Resource):
	@setting_ns.marshal_list_with(device_fields)
	def get(self):
		rows = list_device()
		return [_row_to_dict("tbl_device", r) for r in rows]

	@setting_ns.expect(device_fields)
	def post(self):
		data = request.json or {}
		if "name" not in data:
			return {"error": "'name' is required"}, 400
		try:
			create_device(data)
			return {"success": True}, 201
		except IntegrityError as e:
			return {"error": f"Integrity error: {str(e)}"}, 400
		except Exception as e:
			return {"error": str(e)}, 500


@setting_ns.route("/devices/<int:no>")
class DeviceItem(Resource):
	@setting_ns.marshal_with(device_fields)
	def get(self, no):
		row = read_device(no)
		if not row:
			return {"error": "Not found"}, 404
		return _row_to_dict("tbl_device", row)

	@setting_ns.expect(device_fields)
	def put(self, no):
		data = request.json or {}
		if not data:
			return {"error": "No fields to update"}, 400
		try:
			update_device(no, data)
			row = read_device(no)
			if not row:
				return {"error": "Not found"}, 404
			return _row_to_dict("tbl_device", row)
		except Exception as e:
			return {"error": str(e)}, 500

	def delete(self, no):
		try:
			row = read_device(no)
			if not row:
				return {"error": "Not found"}, 404
			delete_device(no)
			return {"success": True}
		except Exception as e:
			return {"error": str(e)}, 500


