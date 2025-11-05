
;;;;;;;;;;;;
;; 
;; SIMPLE ARM MOVEMENT TASK - first version
;;
;; Hyungseok Oh
;;
;;;;;;;;;;;;
;;
;; Task environment module including ARM MODULE
;;
;;;;;;;;;;;;
;;
;; 2013.07.16
;; ** Initial creation of the module
;;
;;;;;;;;;;;;

;(load-act-r-model "~/Documents/MODEL/usability/1.0/simple_model.lisp")
(load-act-r-model "actr7.x/usability/1.0/model/gait-simple-model.lisp")

(defvar *gait-pos* 0)

(defun set-gait-movement-task-parameters (params)
  (let ((mode (second params)))
    (if mode
	(cond ((string= mode "fixed")
	       (set-parameter-value :set-gait-mode t))
	      ((string= mode "adaptive")
	       (set-parameter-value :set-gait-mode nil)))
	))
  )

(defun init-gait-test ()
  ;(setf *current-time* 0)
  ;(setf *desired-ARM-ANGLE* 0)
  ;(setf *wearable-arm-angle* 0)
  
  ;(setf *response* nil)
  ;(setf *task-num* 0)
  ;(setf *pretrial-time* 0)
  ;(setf *text* nil)
  ;(setf *stroop-list* (make-stroop-list *level*))
					;(setf *WORKLOAD* nil)
  (init-cognitive-global-parameters)

  (setf *gait-pos* 0)
  
  (enable-trace-history)

  (enable-buffer-trace-history)
  (enable-buffer-trace-buffer 'goal)
  (enable-buffer-trace-buffer 'imaginal)
  (enable-buffer-trace-buffer 'manual)
  (enable-buffer-trace-buffer 'production)
  (enable-buffer-trace-buffer 'retrieval)
  (enable-buffer-trace-buffer 'visual)
  (enable-buffer-trace-buffer 'visual-location)
  
  )

(add-act-r-command "init-gait-test" 'init-gait-test)


(defun do-gait-test ()  ;;FUNCTION FOR TEST (DEBUGGING)
  
  (reset)
  
  (init-gait-test)
  
  ;(schedule-event-relative *DELAY* 'next-movement :time-in-ms t)
  
  
  ;(enable-buffer-trace-history)
  ;(enable-buffer-trace-buffer 'goal)
  ;(enable-buffer-trace-buffer 'imaginal)
  ;(enable-buffer-trace-buffer 'manual)
  ;(enable-buffer-trace-buffer 'production)
  ;(enable-buffer-trace-buffer 'retrieval)
  ;(enable-buffer-trace-buffer 'visual)
  ;(enable-buffer-trace-buffer 'visual-location)
    
  (run 4 nil)
  
  )
  

(defun do-gait-test-until-time (time)
  
  ;(reset)
  
  (init-gait-test)

  (let* ((window (open-exp-window "GAIT" :visible nil))
	 (target (add-text-to-exp-window window "O" :x 50 :y 50))
	 (recheck (add-text-to-exp-window window "X" :x 50 :y 75)))

    (install-device window)
    
  ;(enable-buffer-trace-history)
  ;(enable-buffer-trace-buffer 'goal)
  ;(enable-buffer-trace-buffer 'imaginal)
  ;(enable-buffer-trace-buffer 'manual)
  ;(enable-buffer-trace-buffer 'production)
  ;(enable-buffer-trace-buffer 'retrieval)
  ;(enable-buffer-trace-buffer 'visual)
  ;(enable-buffer-trace-buffer 'visual-location)
    
  (run time nil)
  
  (output-ACT-R-model-output-parameter (get-saved-trace))
  (output-learnability-parameter)
  (output-memorability-parameter)
  (output-utility-parameter)
  (output-efficiency-parameter)
  (output-effectiveness-parameter)
  (output-performance-parameter)
  
    ))


