from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
import os
import shutil
import subprocess
import time

from config import SHARED_DIR, PARENT_DIR

from .db import (
    _row_to_dict,
    create_project,
    delete_project,
    list_project,
    read_project,
    update_project,
)


directory_ns = Namespace("directory", path="/api/directory", description="Directory operations")


project_fields = directory_ns.model('Project', {
    'no': fields.Integer(required=True, description='no'),
    'name': fields.String(required=True, description='name'),
    'human': fields.String(description='human'),
    'device': fields.String(description='device'),
    'before_human': fields.String(description='before_human'),
    'before_device': fields.String(description='before_device'),
    'temp': fields.String(description='temp'),
    'last_accessed': fields.String(description='최종 접근 날짜'),
    'reg_date': fields.String(description='등록 날짜'),
})

@directory_ns.route("/action/<string:action_type>")
class PatientAction(Resource):
    def post(self, action_type):
        data = request.json or {}
        patient = data.get("patient", "")
        xml = data.get("xml", "")
        target = os.path.join(SHARED_DIR,f"{patient['name']}", f"{patient['name']}.xml")
        result = 2

        try:
            if action_type == "access":
                try:
                    update_project(patient['no'], {"last_accessed": time.strftime('%Y-%m-%d %H:%M:%S')})
                    with open(os.path.join(PARENT_DIR, "Index.dat"), "w", encoding="utf-8") as f:
                        f.write(f"{patient['name']}\n2\n2")
                except Exception as e:
                    return {"error": f"Failed to write Index.dat: {str(e)}"}, 500
                return {"success": True, "action": action_type}

            elif action_type == "start":
                try:
                    with open(target, "w", encoding="utf-8") as f:
                        f.write(xml)
                    
                    with open(os.path.join(PARENT_DIR, "Index.dat"), "w", encoding="utf-8") as f:
                        f.write(f"{patient['name']}\n1\n2")
                        
                    human = data.get("human", "")
                    device = data.get("device", "")

                    update_project(patient['no'], {"human": human, "device": device})

                except Exception as e:
                    return {"error": f"Failed to write xml: {str(e)}"}, 500
                return {"success": True, "action": action_type}

            elif action_type == "result" or action_type == "display":
                try:
                    with open(os.path.join(PARENT_DIR, "Index.dat"), "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if len(lines) >= 3:
                        result = int(lines[2].strip())
                except Exception as e:
                    return {"error": f"Failed to write Index.dat: {str(e)}"}, 500
                return {"success": True, "action": action_type, "result": result}

            else:
                return {"error": "Invalid action type"}, 400

        except Exception as e:
            return {"error": str(e)}, 500


@directory_ns.route("/project")
class Project(Resource):
    @directory_ns.marshal_list_with(project_fields)
    def get(self):
        rows = list_project()
        result = [_row_to_dict("tbl_project", r) for r in rows]
        return result

    @directory_ns.expect(project_fields)
    def post(self):
        
        data = request.json or {}
        name = data.get("name")
        
        patient = create_project(data)
        
        if not name:
            return {"error": "'name' is required"}, 400
        target = os.path.join(SHARED_DIR, name)
        try:
            os.makedirs(target, exist_ok=True)
            return {"success": True, "patient": patient}
        except Exception as e:
            return {"error": str(e)}, 500
        
        
@directory_ns.route("/project/<int:no>")
class ProjectItem(Resource):
    @directory_ns.marshal_with(project_fields)
    def get(self, no):
        row = read_project(no)
        if not row:
            return {"error": "Not found"}, 404
        return _row_to_dict("tbl_project", row)

    @directory_ns.expect(project_fields)
    def put(self, no):
        data = request.json or {}
        if not data:
            return {"error": "No fields to update"}, 400
        try:
            update_project(no, data)
            row = read_project(no)
            if not row:
                return {"error": "Not found"}, 404
            return _row_to_dict("tbl_project", row)
        except Exception as e:
            return {"error": str(e)}, 500

    def delete(self, no):
        try:
                    
            row = read_project(no)
            if not row:
                return {"error": "Not found"}, 404

            delete_project(no)
            target = os.path.join(SHARED_DIR, row[1])  # Assuming 'name' is the second column

            if os.path.exists(target):
                # remove directory and its contents if necessary
                try:
                    shutil.rmtree(target)
                except Exception:
                    try:
                        os.rmdir(target)
                    except Exception:
                        pass

            return {"success": True}
        except Exception as e:
            return {"error": str(e)}, 500
        
        

@directory_ns.route("/list")
class DirectoryList(Resource):
    def get(self):        
        
        path = request.args.get("path", "")
        target = os.path.join(SHARED_DIR, path) if path else SHARED_DIR

        if not os.path.exists(target):
            return {"error": "Path not found"}, 404

        folders = []
        files = []
        for entry in os.scandir(target):
            if entry.is_dir():
                folders.append({"name": entry.name, "path": entry.path})
            elif entry.is_file():
                files.append({"name": entry.name, "path": entry.path})
        return {"folders": folders, "files": files}
    

@directory_ns.route("/create-folder")
class CreateFolder(Resource):
    def post(self):
        
        data = request.json or {}
        name = data.get("name")
        path = data.get("path", "")
        if not name:
            return {"error": "'name' is required"}, 400
        target = os.path.join(SHARED_DIR, path, name)
        try:
            os.makedirs(target, exist_ok=True)
            return {"success": True, "path": target}
        except Exception as e:
            return {"error": str(e)}, 500


@directory_ns.route("/create-file")
class CreateFile(Resource):
    def post(self):
        
        data = request.json or {}
        name = data.get("name")
        path = data.get("path", "")
        content = data.get("content", "")
        if not name:
            return {"error": "'name' is required"}, 400
        target = os.path.join(SHARED_DIR, path, name)
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": target}
        except Exception as e:
            return {"error": str(e)}, 500
        
        
@directory_ns.route("/upload-xml")
class UploadFile(Resource):
    def post(self):
        
        data = request.json or {}
        filename = data.get("filename")
        xml = data.get("xml")
        
        if not filename:
            return {"error": "'filename' is required"}, 400
        target = os.path.join(SHARED_DIR, filename)
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write(xml)
            return {"success": True, "path": target}
        except Exception as e:
            return {"error": str(e)}, 500


@directory_ns.route("/open-file")
class OpenFile(Resource):
    def get(self):
        path = request.args.get("path", "")
        target = os.path.join(SHARED_DIR, path)
        if not os.path.isfile(target):
            return {"error": "File not found"}, 404

        try:
            open_file_in_vscode(target)
            return {"success": True, "path": target}
        except Exception as e:
            return {"error": str(e)}, 500

def open_file_in_vscode(target):
    try:
        print("기존 VSCode 창 없음 → 새 창 실행")
        subprocess.Popen(["code", "--new-window", target])
        # 창이 뜰 시간 대기
        time.sleep(2)
    except Exception as e:
        print(f"VSCode 실행 오류: {e}")
        raise
        