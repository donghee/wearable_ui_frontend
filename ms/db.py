import os
import sqlite3

from config import BASE_DIR

sql_tbl_project = """ CREATE TABLE IF NOT EXISTS tbl_project (
    no	INTEGER NOT NULL,
    name	TEXT NOT NULL,
    human	TEXT,
    device	TEXT,
    before_human TEXT,
    before_device TEXT,
    temp TEXT,    
    last_accessed text default (DATETIME('now', 'localtime')),
    reg_date text default (DATETIME('now', 'localtime')),

	PRIMARY KEY(no)
); """

sql_tbl_human = """ CREATE TABLE IF NOT EXISTS tbl_human (
	no	INTEGER NOT NULL,
	name	TEXT NOT NULL,
	tags	TEXT,
 
    /* 개인 정보 */
	age	    TEXT,               /* 나이 */
    gender    TEXT,              /* 성별 */
    tendency TEXT,               /* 성향 */
    
    /* Pathology */
    alpha TEXT,               /* alpha */
    beta TEXT,                /* beta */
    gamma TEXT,               /* gamma */
    n TEXT,               /* n */
    rgain TEXT,            /* r-gain */
    fmax TEXT,             /* f-max */
    kpass TEXT,            /* k-pass */
    lopt TEXT,             /* l-opt */
    
    /* 신체 사이즈 */
    upperarm_length TEXT,    /* 상완 길이 */
    upperarm_mass TEXT,      /* 상완 질량 */
    upperarm_location TEXT,    /* 상완 무게중심 위치 */
    forearm_length TEXT,     /* 전완 길이 */
    forearm_mass TEXT,       /* 전완 질량 */
    forearm_location TEXT,    /* 전완 무게중심 위치 */
    arm_circumference TEXT,   /* 팔 둘레 */
    leg_circumference TEXT,   /* 다리 둘레 */
    chest_circumference TEXT, /* 가슴 둘레 */
    belly_circumference TEXT, /* 배 둘레 */
    body_length TEXT,        /* 몸 길이 */
    thigh_length TEXT,      /* 허벅지 길이 */
    calf_length TEXT,       /* 종아리 길이 */
    
    is_default INTEGER DEFAULT 0,
    reg_date text default (DATETIME('now', 'localtime')),

	PRIMARY KEY(no)
); """

sql_tbl_device = """ CREATE TABLE IF NOT EXISTS tbl_device (
    no	INTEGER NOT NULL,
    name	TEXT NOT NULL,
    tags	TEXT,
    
    /* Motor */
    motor_mass TEXT,          /* 모터 질량 */
    motor_angle_upperlimit TEXT, /* 모터 상한 각도 */
    motor_angle_lowerlimit TEXT, /* 모터 하한 각도 */
    motor_damping TEXT,        /* 모터 감쇠 */
    motor_friction TEXT,       /* 모터 마찰 */
    
    /* Velcro */
    velcro_shear_stiffness TEXT,   /* 전단방향 강성 */
    velcro_vertical_stiffness TEXT,  /* 수직 방향 강성 */
    velcro_torsional_stiffness TEXT, /* 비틀림 강성 */
    velcro_strength_steel TEXT,     /* 착용강도 */
    velcro_wear_damping TEXT,        /* 착용댐핑 */
    velcro_wearing_space TEXT,      /* 착용유격 */
    
    /* 사이즈 */    
    upperarm_length TEXT,    /* 상완 길이 */
    upperarm_mass TEXT,      /* 상완 질량 */
    upperarm_location TEXT,    /* 상완 무게중심 위치 */
    forearm_length TEXT,     /* 전완 길이 */
    forearm_mass TEXT,       /* 전완 질량 */
    forearm_location TEXT,    /* 전완 무게중심 위치 */
    shear_direction_wear_error TEXT, /* 전단방향 착용오차 */
    vertical_direction_wear_error TEXT, /* 수직방향 착용오차 */
    torsional_direction_wear_error TEXT, /* 비틀림방향 착용오차 */
    
    is_default INTEGER DEFAULT 0,
    reg_date text default (DATETIME('now', 'localtime')),
    
    PRIMARY KEY(no)
); """

DB_PATH = os.path.join(BASE_DIR, "db_setting.db")

def get_connection():
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'w').close()
    return sqlite3.connect(DB_PATH)

def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql_tbl_project)
    cursor.execute(sql_tbl_human)
    cursor.execute(sql_tbl_device)
    
    
    human = list_human()
    if len(human) == 0:
        sql_insert_human = """INSERT INTO tbl_human (name, tags, age, gender, tendency,
        alpha, beta, gamma, n, rgain, fmax, kpass, lopt,
        upperarm_length, upperarm_mass, upperarm_location,
        forearm_length, forearm_mass, forearm_location,
        arm_circumference, leg_circumference, chest_circumference,
        belly_circumference, body_length, thigh_length, calf_length,
        is_default) 
        VALUES ('기본환자', '기본정보', '70', '1', '100',
        '1', '1', '1', '10', '1', '1', '0.5', '1',
        '0.2817', '1.9783', '42.28',
        '0.2689', '1.1826', '45.74',
        '0.255', '0.3', '0.98',
        '0.879', '0.86', '0.359', '0.405',
        1)"""
        cursor.execute(sql_insert_human)

    device = list_device()
    if len(device) == 0:
        sql_insert_device = """INSERT INTO tbl_device (name, tags, motor_mass, motor_angle_upperlimit, motor_angle_lowerlimit, motor_damping, motor_friction,
        velcro_shear_stiffness, velcro_vertical_stiffness, velcro_torsional_stiffness, velcro_strength_steel, velcro_wear_damping, velcro_wearing_space,
        upperarm_length, upperarm_mass, upperarm_location, forearm_length, forearm_mass, forearm_location,
        shear_direction_wear_error, vertical_direction_wear_error, torsional_direction_wear_error,
        is_default) VALUES ('기본디바이스', '기본정보', '1', '-2.2689', '-0.1745', '2', '15',
        '1700', '5200', '18.72', '100000', '13', '0.001',
        '0.2', '2', '50', '0.2', '2', '50',
        '0.02', '0.02', '10',
        1)"""
        cursor.execute(sql_insert_device)

    conn.commit()
    conn.close()
    

def _get_columns(table_name: str):
	"""Return column names for the given table using PRAGMA."""
	conn = get_connection()
	try:
		cur = conn.cursor()
		cur.execute(f"PRAGMA table_info({table_name})")
		cols = [row[1] for row in cur.fetchall()]  # [ (cid, name, type, ...), ... ]
		return cols
	finally:
		conn.close()


def _row_to_dict(table_name: str, row):
	if row is None:
		return None
	cols = _get_columns(table_name)
	return {k: v for k, v in zip(cols, row)}

    
# CRUD for tbl_project
def create_project(data):
    conn = get_connection()
    cursor = conn.cursor()
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    sql = f"INSERT INTO tbl_project ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, tuple(data.values()))
    
    last_id = cursor.lastrowid
    cursor.execute("SELECT * FROM tbl_project WHERE no=?", (last_id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return _row_to_dict('tbl_project', row)

def list_project():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_project order by last_accessed desc")
    results = cursor.fetchall()
    conn.close()
    return results

def read_project(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_project WHERE no=?", (no,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_project(no, data):
    conn = get_connection()
    cursor = conn.cursor()
    assignments = ', '.join([f"{k}=?" for k in data.keys()])
    sql = f"UPDATE tbl_project SET {assignments} WHERE no=?"
    cursor.execute(sql, tuple(data.values()) + (no,))
    conn.commit()
    conn.close()

def delete_project(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tbl_project WHERE no=?", (no,))
    conn.commit()
    conn.close()

# CRUD for tbl_human
def create_human(data):
    conn = get_connection()
    cursor = conn.cursor()
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    sql = f"INSERT INTO tbl_human ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, tuple(data.values()))
    conn.commit()
    conn.close()
    
def list_human():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_human order by case when is_default = 1 then 0 else 1 end, no desc")
    results = cursor.fetchall()
    conn.close()
    return results

def read_human(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_human WHERE no=?", (no,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_human(no, data):
    conn = get_connection()
    cursor = conn.cursor()
    assignments = ', '.join([f"{k}=?" for k in data.keys()])
    sql = f"UPDATE tbl_human SET {assignments} WHERE no=?"
    cursor.execute(sql, tuple(data.values()) + (no,))
    conn.commit()
    conn.close()

def delete_human(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tbl_human WHERE no=?", (no,))
    conn.commit()
    conn.close()

# CRUD for tbl_device
def create_device(data):
    conn = get_connection()
    cursor = conn.cursor()
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    sql = f"INSERT INTO tbl_device ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, tuple(data.values()))
    conn.commit()
    conn.close()
    
def list_device():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_device order by case when is_default = 1 then 0 else 1 end, no desc")
    results = cursor.fetchall()
    conn.close()
    return results

def read_device(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tbl_device WHERE no=?", (no,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_device(no, data):
    conn = get_connection()
    cursor = conn.cursor()
    assignments = ', '.join([f"{k}=?" for k in data.keys()])
    sql = f"UPDATE tbl_device SET {assignments} WHERE no=?"
    cursor.execute(sql, tuple(data.values()) + (no,))
    conn.commit()
    conn.close()

def delete_device(no):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tbl_device WHERE no=?", (no,))
    conn.commit()
    conn.close()
