
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

(load-act-r-model "~/Documents/MODEL/usability/1.0/simple_model.lisp")


(defun set-arm-movement-task-parameters (params)
  (let ((mode (second params)))
    (if mode
	(cond ((string= mode "fixed")
	       (set-parameter-value :set-arm-mode t))
	      ((string= mode "adaptive")
	       (set-parameter-value :set-arm-mode nil)))
	))
  )

(defun init-test ()
  
  (init-model-parameters)
  
  ;(setf *current-time* 0)
  ;(setf *desired-ARM-ANGLE* 0)
  ;(setf *wearable-arm-angle* 0)
  
  ;(setf *response* nil)
  ;(setf *task-num* 0)
  ;(setf *pretrial-time* 0)
  ;(setf *text* nil)
  ;(setf *stroop-list* (make-stroop-list *level*))
					;(setf *WORKLOAD* nil)
  
  (enable-trace-history)
  )


(defun do-test ()  ;;FUNCTION FOR TEST (DEBUGGING)
  
  (reset)
  
  (init-test)
  
  ;(schedule-event-relative *DELAY* 'next-movement :time-in-ms t)
  
  
  (enable-buffer-trace-history)
  (enable-buffer-trace-buffer 'goal)
  (enable-buffer-trace-buffer 'imaginal)
  (enable-buffer-trace-buffer 'manual)
  (enable-buffer-trace-buffer 'production)
  (enable-buffer-trace-buffer 'retrieval)
  (enable-buffer-trace-buffer 'visual)
  (enable-buffer-trace-buffer 'visual-location)
    
  (run 4 nil)
  
  )
  

(defun do-test-until-time (time)
  
  ;(reset)
  
  (init-test)
  
  ;(schedule-event-relative *DELAY* 'next-movement :time-in-ms t)
  
  
  (enable-buffer-trace-history)
  (enable-buffer-trace-buffer 'goal)
  (enable-buffer-trace-buffer 'imaginal)
  (enable-buffer-trace-buffer 'manual)
  (enable-buffer-trace-buffer 'production)
  (enable-buffer-trace-buffer 'retrieval)
  (enable-buffer-trace-buffer 'visual)
  (enable-buffer-trace-buffer 'visual-location)
    
  (run time nil)
  
  (output-ACT-R-model-output-parameter (get-saved-trace))
  (output-learnability-parameter)
  (output-memorability-parameter)
  (output-utility-parameter)
  (output-efficiency-parameter)
  (output-effectiveness-parameter)
  (output-performance-parameter)
  
  )
