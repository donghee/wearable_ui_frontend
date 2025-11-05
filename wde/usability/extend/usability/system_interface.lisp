
;;;;;;;;;;;;
;; 
;; SYSTEM INTERFACE MODULE - first version
;;
;; Hyungseok Oh
;;
;;;;;;;;;;;;
;;
;; This module processes the system interface including input/output parameters in the digital human twin
;;
;;;;;;;;;;;;
;;
;; 2013.07.16
;; ** Initial creation of the module
;;
;;;;;;;;;;;;

;(actr-load "ACT-R:usability;1.0;model_parameters.lisp")
;(actr-load "ACT-R:usability;1.0;model;simple_model.lisp")
;(actr-load "ACT-R:usability;1.0;task;simple_arm_movement_task.lisp")

;(print (sb-posix:getcwd))
(load "wde/usability/extend/usability/1.0/model_parameters.lisp")
;(load "actr7.x/usability/1.0/model/simple_model.lisp")
;(load "actr7.x/usability/1.0/task/simple_arm_movement_task.lisp")
;(load "actr7.x/usability/1.0/task/simple_gait_task.lisp")

;;;
;;; Run the simulation model
;;;
(defun run-the-model-until-time (time task age cog-level trajectory force noise)  
                              ;; time -- seconds 
                              ;; task - list 2 params -- "arm" // "adaptive" or "fixed"
                              ;; age  -- twin age
                              ;; cog-level -- "high" "low"
                              ;; trajectory -- list 2 params -- peak, angular-velocity
                              ;; force -- percentage value
                              ;; noise -- individual difference value - under 10
  
  (let ((total-usability-value 0)
	(learnability     0)
	(memorability     0) ;do not use yet
	(utility          0) 
	(efficiency       0) ;mental workload
	(effectiveness    0) ;human error
	(performance-time 0)
	(error-value      0)
	(model-output-text))

    (reset)

    (when (eq trajectory 0) (setf trajectory nil))

    ;;load task and model
    (cond ((string= (first task) "arm")
	   (load "wde/usability/extend/usability/1.0/task/simple_arm_movement_task.lisp"))
	  ((string= (first task) "gait")
	   (load "wde/usability/extend/usability/1.0/task/simple_gait_task.lisp")))
  
  ;;;setting parameters
  (input-twin-age-parameter age)
  (input-task-information-parameter task)
  (input-twin-cognitive-ability-parameter cog-level)
  (input-device-force-timeline-parameter force)
    (input-twin-noise-parameter noise)
    (input-task-performance-time-parameter time)
  
  (setting-twin-parameters-for-model *SETTING-INPUT-PARAMETERS*)
   
  ;;;run the model
  (cond ((string= (first task) "arm") (setf model-output-text (do-test-until-time time)))
	((string= (first task) "gait") (setf model-output-text (do-gait-test-until-time time))))
      
  
  ;;;calculate the simulation output parameters
  (setf learnability  (get-value-learnability-parameter))
  (setf memorability  (get-value-memorability-parameter))
  (setf utility       (get-value-utility-parameter))
  (setf efficiency    (get-value-mental-workload-parameter))
  (setf effectiveness (get-value-effectiveness-parameter))
  (setf performance-time 0) ;;fix it!!
  
  (setf total-usability-value 
    (/ (+ learnability utility efficiency effectiveness) 4))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;     RETURN VALUE
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; 1. Total Usability value
  ;; 2. LIST - Learnability, Memorability, Efficiency, Effectiveness
  ;; 3. Performance time
  ;; 4. Error value
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (list (floor total-usability-value)
	(list learnability utility efficiency effectiveness) 
	performance-time 
	(get-value-ACT-R-model-output))
    ))

(add-act-r-command "run-the-model-until-time" 'run-the-model-until-time "run-the-model-until-time function")



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; INPUT/OUTPUT PARAMETERS STRUCTURE
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;set-input-parameters
(defstruct setting-input-parameter-list
  twin-age
  twin-gender
  twin-cognitive-ability
  twin-cognitive-focus-ability
  twin-physical-ability
  twin-physical-agility
  task-information
  twin-trajectory-timeline
  device-force-timeline
  event-reaction-time-list
  event-error-list
  task-performance-time
  task-performance-reserve
  twin-noise
  )

(defvar *SETTING-INPUT-PARAMETERS* (make-setting-input-parameter-list))

;;;set-output-variables
(defstruct setting-output-list
  ACT-R-output
  human-error
  mental-workload
  learnability
  memorability
  utility
  effectiveness
  total-usability
  )

(defvar *OUTPUT-PARAMETERS* (make-setting-output-list))


(defun setting-input-parameters ()
  (setting-twin-parameters-for-model *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "setting-input-parameters" 'setting-input-parameters)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; ** TWIN -> MODEL SYSTEM ** INPUT PARAMETER FUNCTIONS
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; TWIN - PERSONAL INFORMATION
;;;

;;;
;;; TWIN AGE
;;;
(defun input-twin-age-parameter (value)
  (setf (setting-input-parameter-list-twin-age *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-age ()
  (setting-input-parameter-list-twin-age *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-age-parameter" 'input-twin-age-parameter)
(add-act-r-command "get-value-twin-age" 'get-value-twin-age)

;;;
;;; TWIN GENDER
;;;
(defun input-twin-gender-parameter (value)
  (setf (setting-input-parameter-list-twin-gender *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-gender ()
  (setting-input-parameter-list-twin-gender *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-gender-parameter" 'input-twin-age-parameter)
(add-act-r-command "get-value-twin-gender" 'get-value-twin-gender)


;;;
;;; TWIN COGNITIVE ABILITY
;;;
(defun input-twin-cognitive-ability-parameter (value)
  (setf (setting-input-parameter-list-twin-cognitive-ability *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-cognitive-ability ()
  (setting-input-parameter-list-twin-cognitive-ability *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-cognitive-ability-parameter" 'input-twin-cognitive-ability-parameter)
(add-act-r-command "get-value-twin-cognitive-ability" 'get-value-twin-cognitive-ability)


;;;
;;; TWIN COGNITIVE FOCUS ABILITY
;;;
(defun input-twin-cognitive-focus-ability-parameter (value)
  (setf (setting-input-parameter-list-twin-cognitive-focus-ability *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-cognitive-focus-ability ()
  (setting-input-parameter-list-twin-cognitive-focus-ability *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-cognitive-focus-ability-parameter" 'input-twin-cognitive-focus-ability-parameter)
(add-act-r-command "get-value-twin-cognitive-focus-ability" 'get-value-twin-cognitive-focus-ability)


;;;
;;; TWIN PHYSICAL ABILITY (POWER)
;;;
(defun input-twin-physical-ability-parameter (value)
  (setf (setting-input-parameter-list-twin-physical-ability *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-physical-ability ()
  (setting-input-parameter-list-twin-physical-ability *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-physical-ability-parameter" 'input-twin-physical-ability-parameter)
(add-act-r-command "get-value-twin-physical-ability" 'get-value-twin-physical-ability)


;;;
;;; TWIN PHYSICAL AGILITY (AGILITY, SPEED)
;;;
(defun input-twin-physical-agility-parameter (value)
  (setf (setting-input-parameter-list-twin-physical-agility *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-physical-agility ()
  (setting-input-parameter-list-twin-physical-agility *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-physical-agility-parameter" 'input-twin-physical-agility-parameter)
(add-act-r-command "get-value-twin-physical-agility" 'get-value-twin-physical-agility)


;;;
;;; TWIN - TASK INFORMATION
;;;

;;;
;;; TASK INFORMATION
;;;
(defun input-task-information-parameter (value)
  (setf (setting-input-parameter-list-task-information *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-task-information ()
  (setting-input-parameter-list-task-information *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-task-information-parameter" 'input-task-information-parameter)
(add-act-r-command "get-value-task-information" 'get-value-task-information)


;;;
;;; TWIN - TASK PERFORMANCE INFORMATION
;;;

;;;
;;; TASK RESULTS 1 (TRAJECTORY TIMELINE)
;;;
(defun input-twin-trajectory-timeline-parameter (value)
  (setf (setting-input-parameter-list-twin-trajectory-timeline *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-trajectory-timeline ()
  (setting-input-parameter-list-twin-trajectory-timeline *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-trajectory-timeline-parameter" 'input-twin-trajectory-timeline-parameter)
(add-act-r-command "get-value-twin-trajectory-timeline" 'get-value-twin-trajectory-timeline)


;;;
;;; TASK RESULTS 2 (DEVICE FORCE TIMELINE)
;;;
(defun input-device-force-timeline-parameter (value)
  (setf (setting-input-parameter-list-device-force-timeline *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-device-force-timeline ()
  (setting-input-parameter-list-device-force-timeline *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-device-force-timeline-parameter" 'input-device-force-timeline-parameter)
(add-act-r-command "get-value-device-force-timeline" 'get-value-device-force-timeline)


;;;
;;; TASK RESULTS - RESPONSE TIME PER TASK EVENT (LIST EVENT RESPONSE-TIME)
;;;
(defun input-event-reaction-time-list-parameter (value)
  (setf (setting-input-parameter-list-event-reaction-time-list *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-event-reaction-time-list ()
  (setting-input-parameter-list-event-reaction-time-list *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-event-reaction-time-list-parameter" 'input-event-reaction-time-list-parameter)
(add-act-r-command "get-value-event-reaction-time-list" 'get-value-event-reaction-time-list)


;;;
;;; TASK RESULTS - ERROR PER TASK EVENT (LIST EVENT ERROR)
;;;
(defun input-event-error-list-parameter (value)
  (setf (setting-input-parameter-list-event-error-list *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-event-error-list ()
  (setting-input-parameter-list-event-error-list *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-event-error-list-parameter" 'input-event-error-list-parameter)
(add-act-r-command "get-value-event-error-list" 'get-value-event-error-list)


;;;
;;; TASK RESULTS - PERFORMANCE TIME
;;;
(defun input-task-performance-time-parameter (value)
  (setf (setting-input-parameter-list-task-performance-time *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-task-performance-time ()
  (setting-input-parameter-list-task-performance-time *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-task-performance-time-parameter" 'input-task-performance-time-parameter)
(add-act-r-command "get-value-task-performance-time" 'get-value-task-performance-time)

;;;
;;; RESERVE PARAMETER
;;;
(defun input-task-performance-reserve-parameter (value)
  (setf (setting-input-parameter-list-task-performance-reserve *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-task-performance-reserve ()
  (setting-input-parameter-list-task-performance-reserve *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-task-performance-reserve-parameter" 'input-task-performance-reserve-parameter)
(add-act-r-command "get-value-task-performance-reserve" 'get-value-task-performance-reserve)


;;;
;;; TWIN NOISE PARAMETER
;;;
(defun input-twin-noise-parameter (value)
  (setf (setting-input-parameter-list-twin-noise *SETTING-INPUT-PARAMETERS*) value))

(defun get-value-twin-noise ()
  (setting-input-parameter-list-twin-noise *SETTING-INPUT-PARAMETERS*))

(add-act-r-command "input-twin-noise-parameter" 'input-twin-noise-parameter)
(add-act-r-command "get-value-twin-noise" 'get-value-twin-noise)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; ** MODEL -> TWIN SYSTEM ** OUTPUT PARAMETER FUNCTIONS
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; ACT-R MODEL OUTPUT
;;;
(defun input-ACT-R-model-output (value)
  (setf (setting-output-list-ACT-R-output *OUTPUT-PARAMETERS*) value))

(defun get-value-ACT-R-model-output ()
  (setting-output-list-ACT-R-output *OUTPUT-PARAMETERS*))

(add-act-r-command "input-ACT-R-model-output" 'input-ACT-R-model-output)
(add-act-r-command "get-value-ACT-R-model-output" 'get-value-ACT-R-model-output)



;;;
;;; HUMAN ERROR
;;;
(defun input-human-error-parameter (value)
  (setf (setting-output-list-human-error *OUTPUT-PARAMETERS*) value))

(defun get-value-human-error-parameter ()
  (setting-output-list-human-error *OUTPUT-PARAMETERS*))

(add-act-r-command "input-human-error-parameter" 'input-human-error-parameter)
(add-act-r-command "get-value-human-error-parameter" 'get-value-human-error-parameter)



;;;
;;; MENTAL WORKLOAD
;;;
(defun input-mental-workload-parameter (value)
  (setf (setting-output-list-mental-workload *OUTPUT-PARAMETERS*) value))

(defun get-value-mental-workload-parameter ()
  (setting-output-list-mental-workload *OUTPUT-PARAMETERS*))

(add-act-r-command "input-mental-workload-parameter" 'input-mental-workload-parameter)
(add-act-r-command "get-value-mental-workload-parameter" 'get-value-mental-workload-parameter)



;;;
;;; LEARNABILITY
;;;
(defun input-learnability-parameter (value)
  (setf (setting-output-list-learnability *OUTPUT-PARAMETERS*) value))

(defun get-value-learnability-parameter ()
  (setting-output-list-learnability *OUTPUT-PARAMETERS*))

(add-act-r-command "input-learnability-parameter" 'input-learnability-parameter)
(add-act-r-command "get-value-learnability-parameter" 'get-value-learnability-parameter)


;;;
;;; MEMORABILITY
;;;
(defun input-memorability-parameter (value)
  (setf (setting-output-list-memorability *OUTPUT-PARAMETERS*) value))

(defun get-value-memorability-parameter ()
  (setting-output-list-memorability *OUTPUT-PARAMETERS*))

(add-act-r-command "input-memorability-parameter" 'input-memorability-parameter)
(add-act-r-command "get-value-memorability-parameter" 'get-value-memorability-parameter)



;;;
;;; UTILITY
;;;
(defun input-utility-parameter (value)
  (setf (setting-output-list-utility *OUTPUT-PARAMETERS*) value))

(defun get-value-utility-parameter ()
  (setting-output-list-utility *OUTPUT-PARAMETERS*))

(add-act-r-command "input-utility-parameter" 'input-utility-parameter)
(add-act-r-command "get-value-utility-parameter" 'get-value-utility-parameter)



;;;
;;; EFFECTIVENESS
;;;
(defun input-effectiveness-parameter (value)
  (setf (setting-output-list-utility *OUTPUT-PARAMETERS*) value))

(defun get-value-effectiveness-parameter ()
  (setting-output-list-utility *OUTPUT-PARAMETERS*))

(add-act-r-command "input-effectiveness-parameter" 'input-effectiveness-parameter)
(add-act-r-command "get-value-effectiveness-parameter" 'get-value-effectiveness-parameter)



;;;
;;; TOTAL USABILITY VALUE
;;;
(defun input-total-usability-parameter (value)
  (setf (setting-output-list-total-usability *OUTPUT-PARAMETERS*) value))

(defun get-value-total-usability-parameter ()
  (setting-output-list-total-usability *OUTPUT-PARAMETERS*))

(add-act-r-command "input-total-usability-parameter" 'input-total-usability-parameter)
(add-act-r-command "get-value-total-usability-parameter" 'get-value-total-usability-parameter)

