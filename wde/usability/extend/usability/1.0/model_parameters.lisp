
;;;;;;;;;;;;
;; 
;; INPUT MODEL PARAMETERS MODULE - first version
;;
;; Hyungseok Oh
;;
;;;;;;;;;;;;
;;
;; This module processes input values in the digital human twin
;;
;;;;;;;;;;;;
;;
;; 2013.07.16
;; ** Initial creation of the module
;;
;;;;;;;;;;;;

;;;;;;;;;;;;
;;;
;;; Input model parameters 
;;;
;;;;;;;;;;;;

(defun setting-twin-parameters-for-model (setting)
  ;;;
  ;;; Setting Model Parameters
  ;;;
  (let ((age (get-value-twin-age))
	(cognitive-ability (get-value-twin-cognitive-ability))
	(task-inform (get-value-task-information))
	(perform-time (get-value-task-performance-time))
	(error-rate (get-value-device-force-timeline)))
  
    ;;;latency factor
    (if cognitive-ability
	(cond ((equal cognitive-ability "high") (set-parameter-value :lf 0.1))
	      ((equal cognitive-ability "low")  (set-parameter-value :lf 1.0)))
	(set-parameter-value :lf 0.1)) ;;defalut value

    ;;;age
    (if age
	(if (> age 80) (set-parameter-value :lf 1.0)
	    (if (> age 60) (set-parameter-value :lf 0.5)
		(if (> age 30) (set-parameter-value :lf 0.2)
		    (if (> age 20) (set-parameter-value :lf 0.1)
		    (set-parameter-value :lf 0.2)))))
	(set-parameter-value :lf 0.1))
    
    
    ;;;task information
    (when (> (list-length task-inform) 0)
      (cond ((string= (first task-inform) "arm")
	     (set-arm-movement-task-parameters task-inform))
	    ((string= (first task-inform) "gait")
	     (set-gait-movement-task-parameters task-inform))))

    ;;;perform-time (gait)
    (if perform-time
	(set-parameter-value :set-data-for-perform-time perform-time)
	(set-parameter-value :set-data-for-perform-time 10))

    ;;;error-rate (gait)
    (if perform-time
	(set-parameter-value :set-gait-force-rate error-rate)
	(set-parameter-value :set-gait-force-rate 20))

    
    ))

(add-act-r-command "setting-twin-parameters-for-model" 'setting-twin-parameters-for-model)


;;;;;;;;;;;;
;;;
;;; Output model parameters 
;;;
;;;;;;;;;;;;

(defun output-performance-parameter ()
  (let ((value 0))
    value
  ))

(defun output-ACT-R-model-output-parameter (trace)
  (let ((result nil))
    (dolist (x trace)
      ;(setf result (append result (list (fourth x)))))
      (setf result (append result (list (split-by-space (fourth x))))))
    
    (input-ACT-R-model-output result)
    ))

;;;
;;; PARSING STRING FUNCTION
;;;
 (defun parsing-helper (input-string)
  (let ((start (position #\   input-string)))
    (if start
        (let ((end (position #\  input-string :start (+ start 1))))
          (if end
              (cons (subseq input-string (+ start 1) end)
                    (split-by-space (subseq input-string end)))
            (list (subseq input-string (+ start 1)))))
      nil)))

 (defun split-by-space (input-string)
  (let ((start (position #\  input-string)))
    (if (and start (> start 0))
        (remove-if (lambda (s) (string-equal s "")) (parsing-helper (concatenate 'string " " input-string)))
      (remove-if (lambda (s) (string-equal s "")) (parsing-helper input-string)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LEARNABILITY - ReControl rate
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-learnability-parameter ()
  (let ((value 0)
	(output-list (get-value-ACT-R-model-output))
	(continuation 0)
	(recontrol 0))
    
    (dolist (output output-list)
      (when (equal (third output) "PRODUCTION-FIRED")
	(cond ((equal (fourth output) "MOVEMENT-CONTINUATION") (incf continuation 1))
	      ((equal (fourth output) "MOVEMENT-RE-CONTROL") (incf recontrol 1)))))

    (if (= (+ continuation recontrol) 0) (setf value 1)
	(setf value (floor (* 100 (/ continuation (+ continuation recontrol))))))
    
    (input-learnability-parameter value)
  ))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; MEMORABILITY - NOT YET!!
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-memorability-parameter ()
  (let ((value 0))
    (input-memorability-parameter value)
  ))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; UTILITY - satisfaction??
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-utility-parameter ()
  (let ((value 0)
	(main-task nil))
    
    (cond ((equal (first (get-value-task-information)) "arm") (setf main-task 'arm))
	  ((equal (first (get-value-task-information)) "gait") (setf main-task 'gait))
	  )
    
    (setf value (get-total-utility-no-decay main-task))
    (input-utility-parameter value)
    ))

(defparameter *UTILITY* nil)
(defparameter *SAVE-UTILITY* t)

(defun get-subtask-utility ()
  (let ((pb (make-workload :buffer 'production :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0))
        (vb (make-workload :buffer 'visual     :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0))
        (rb (make-workload :buffer 'retrieval  :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (ib (make-workload :buffer 'imaginal   :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (mb (make-workload :buffer 'manual     :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0))
        (task nil)
        (pre-task nil))

    (dolist (current-buffer-record (get-current-buffer-trace)) ;;buffer-record every mstime
      (let* ((mstime       (buffer-record-ms-time current-buffer-record))  
             (buffers      (buffer-record-buffers current-buffer-record))
             (goal-buffer  (nth 6 buffers)) ;;goal buffer is 2nd buffer of buffer-records
             (goal-chunk   (buffer-summary-chunk-name goal-buffer))
             (goal-request (buffer-summary-request goal-buffer)))

        (when (and goal-chunk (get-chunk (read-from-string goal-chunk)))
          (setf task 
                (cdr (assoc 'TASK (act-r-chunk-slot-value-lists (get-chunk (read-from-string goal-chunk)))))))
        
        (when goal-request
          (setf task
                (read-from-string (second (member "TASK" (split-by-space goal-request) :test #'equal)))))

        (when (and pre-task (not (eq task pre-task))) ;;TASK CHANGED
          (let ((value  (+ (workload-value pb) 
                           (workload-value vb) 
                           (workload-value rb) 
                           (workload-value ib) 
                           (workload-value mb))))
            ;(format t "~%TASK: ~a // VALUE: ~f" task value)
            (push (make-workload-reference :value value :task pre-task :time (/ mstime 1000)) *UTILITY*))
          
          ;;workload initiation
          (setf pb (make-workload :buffer 'production :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0))
          (setf vb (make-workload :buffer 'visual     :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0))
          (setf rb (make-workload :buffer 'retrieval  :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
          (setf ib (make-workload :buffer 'imaginal   :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
          (setf mb (make-workload :buffer 'manual     :value 0 :weight 0 :active nil :s-time 0.0 :e-time 0.0)))

        (setf pre-task task)
          
        (dolist (buffer buffers)
          (case (buffer-summary-name buffer)
              ((production) (get-buffer-workload pb buffer mstime *SAVE-UTILITY*))
              ((visual)     (get-buffer-workload vb buffer mstime *SAVE-UTILITY*))
              ((retrieval)  (get-buffer-workload rb buffer mstime *SAVE-UTILITY*))
              ((imaginal)   (get-buffer-workload ib buffer mstime *SAVE-UTILITY*))
              ((manual)     (get-buffer-workload mb buffer mstime *SAVE-UTILITY*))))))

     (let ((value  (+ (workload-value pb) 
                      (workload-value vb) 
                      (workload-value rb) 
                      (workload-value ib) 
                      (workload-value mb))))
       (push (make-workload-reference :value value :task task :time  (mp-time)) *UTILITY*))))

(defun get-total-utility-no-decay (task)
  (let ((total 0)
	(ctime (mp-time)))
    
    (get-subtask-utility)
    
    (dolist (ref *UTILITY*)
      (let* ((ctask (workload-reference-task ref))
             (rtime (- ctime (workload-reference-time ref)))
             (wv    (workload-reference-value ref))
             )
	(incf total wv)
	))
    
    (floor (* 100 (/ (- (* 2 (mp-time)) total) (* 2 (mp-time)))))
    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; EFFICIENCY - MENTAL WORKLOAD
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-efficiency-parameter ()
  (let ((value 0)
	(main-task nil))
    
    (cond ((equal (first (get-value-task-information)) "arm") (setf main-task 'arm))
	  ((equal (first (get-value-task-information)) "gait") (setf main-task 'gait))
	  )
    
    (setf value (get-total-workload main-task))
    
    (input-mental-workload-parameter value)
    ))


;;;
;;; structure of workload
;;;
(defparameter *PRIMARY-GOAL* "FIRST-GOAL") ;;Input the name of the primary goal this parameter

(defparameter *SAVE-WORKLOAD* nil)
(defparameter *SCALE* 1.0)

(defparameter *SAVE-WORKLOAD* t)

(defparameter *WORKLOAD* nil)
(defparameter *WORKLOAD-LIST* nil)


(defparameter *THRESHOLD-WORKLOAD* 1.0)


(defstruct workload-reference
  value
  task
  time)
  
(defstruct workload 
  buffer
  value
  weight
  active
  s-time
  e-time)

(defun get-buffer-workload (buffer record mstime save)
  (cond ((and save 
              (not (workload-active buffer)) 
              (or (buffer-summary-busy record) (buffer-summary-request record)))
         (setf (workload-active buffer) t)
         (setf (workload-s-time buffer) (/ mstime 1000)))
        ((and (workload-active buffer) 
              (or (not (buffer-summary-busy record)) (buffer-summary-busy->free record) (buffer-summary-error record)))
         (setf (workload-active buffer) nil)
         (setf (workload-e-time buffer) (/ mstime 1000))
         (setf (workload-value buffer) (+ (workload-value buffer)
                                          (* (workload-weight buffer) 
                                             (- (workload-e-time buffer) (workload-s-time buffer))))))))


(defun get-subtask-workload ()
  (let ((pb (make-workload :buffer 'production :value 0 :weight 2 :active nil :s-time 0.0 :e-time 0.0))
        (vb (make-workload :buffer 'visual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (rb (make-workload :buffer 'retrieval  :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0))
        (ib (make-workload :buffer 'imaginal   :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0))
        (mb (make-workload :buffer 'manual     :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0))
        (task nil)
        (pre-task nil))

    (dolist (current-buffer-record (get-current-buffer-trace)) ;;buffer-record every mstime
      (let* ((mstime       (buffer-record-ms-time current-buffer-record))  
             (buffers      (buffer-record-buffers current-buffer-record))
             (goal-buffer  (nth 6 buffers)) ;;goal buffer is 2nd buffer of buffer-records
             (goal-chunk   (buffer-summary-chunk-name goal-buffer))
             (goal-request (buffer-summary-request goal-buffer)))

        (when (and goal-chunk (get-chunk (read-from-string goal-chunk)))
          (setf task 
                (cdr (assoc 'TASK (act-r-chunk-slot-value-lists (get-chunk (read-from-string goal-chunk)))))))
        
        (when goal-request
          (setf task
                (read-from-string (second (member "TASK" (split-by-space goal-request) :test #'equal)))))

        (when (and pre-task (not (eq task pre-task))) ;;TASK CHANGED
          (let ((value  (+ (workload-value pb) 
                           (workload-value vb) 
                           (workload-value rb) 
                           (workload-value ib) 
                           (workload-value mb))))
            ;(format t "~%TASK: ~a // VALUE: ~f" task value)
            (push (make-workload-reference :value value :task pre-task :time (/ mstime 1000)) *WORKLOAD*))
          
          ;;workload initiation
          (setf pb (make-workload :buffer 'production :value 0 :weight 2 :active nil :s-time 0.0 :e-time 0.0))
          (setf vb (make-workload :buffer 'visual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
          (setf rb (make-workload :buffer 'retrieval  :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0))
          (setf ib (make-workload :buffer 'imaginal   :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0))
          (setf mb (make-workload :buffer 'manual     :value 0 :weight 4 :active nil :s-time 0.0 :e-time 0.0)))

        (setf pre-task task)
          
        (dolist (buffer buffers)
          (case (buffer-summary-name buffer)
              ((production) (get-buffer-workload pb buffer mstime *SAVE-WORKLOAD*))
              ((visual)     (get-buffer-workload vb buffer mstime *SAVE-WORKLOAD*))
              ((retrieval)  (get-buffer-workload rb buffer mstime *SAVE-WORKLOAD*))
              ((imaginal)   (get-buffer-workload ib buffer mstime *SAVE-WORKLOAD*))
              ((manual)     (get-buffer-workload mb buffer mstime *SAVE-WORKLOAD*))))))

     (let ((value  (+ (workload-value pb) 
                      (workload-value vb) 
                      (workload-value rb) 
                      (workload-value ib) 
                      (workload-value mb))))
       (push (make-workload-reference :value value :task task :time  (mp-time)) *WORKLOAD*))))

             
;;;
;;; DECAY FUNCTION
;;;
(defparameter *decay* .5);0.156);.6)

(defun get-workload-including-decay (task reference decay ctime) ;;(WORKLOAD-DELAY 'MEMORY *WORKLOAD* (mp-time))
  (let ((total 0))
    (dolist (ref reference)
      (let* ((ctask (workload-reference-task ref))
             (rtime (- ctime (workload-reference-time ref)))
             (wv    (workload-reference-value ref))
             (tv    (if (= rtime 0) wv (* wv (expt rtime (* decay -1))))))
        ;(when (eq ctask task) (incf total tv))
	(incf total tv)
	))
    total
    ))

(defun get-workload-no-decay (task reference ctime)
  (let ((total 0))
    (dolist (ref reference)
      (let* ((ctask (workload-reference-task ref))
             (rtime (- ctime (workload-reference-time ref)))
             (wv    (workload-reference-value ref))
             )
	(incf total wv)
	))
    total
    ))

;;;
;;; TOTAL WORKLOAD
;;;
(defun get-total-workload (task)
  (get-subtask-workload)
  ;(get-workload-including-decay task *WORKLOAD* *decay* (mp-time))
  (floor (* 100 (/ (get-workload-no-decay task *WORKLOAD* (mp-time)) (* 15 (mp-time)))))
  )      

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; EFFECTIVENESS - HUMAN ERROR
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-effectiveness-parameter ()
  (let ((value 0)
	(force (get-value-device-force-timeline)))
    
    (setf value (+ force (act-r-noise (get-value-twin-noise))))
    (input-human-error-parameter value)
    ))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; TOTAL
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun output-total-model-parameters ()
  (output-ACT-R-model-output-parameter (get-saved-trace))
  (output-learnability-parameter)
  (output-memorability-parameter)
  (output-utility-parameter)
  (output-efficiency-parameter)
  (output-effectiveness-parameter)
  (output-performance-parameter)
  )

(add-act-r-command "output-total-model-parameters" 'output-total-model-parameters)

  




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; BRAIN ACTIVITY FUNCTION!
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; 0. Total value
;; 1-1. Imaginal
;; 1-2. Visual
;; 1-3. Production
;; 1-4. Declarative
;; 2. Brain activity Time Series
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun output-brain-activity-parameter ()
  (let ((value 0)
	(main-task nil))
    
    (cond ((equal (first (get-value-task-information)) "arm") (setf main-task 'arm))
	  ((equal (first (get-value-task-information)) "gait") (setf main-task 'gait))
	  )
    
    (setf value (get-total-brain-activity))
    
    ;(input-mental-workload-parameter value)
    ))


(defstruct brain-activity-reference
  total-value
  production
  visual
  retrieval
  imaginal
  manual
  time)
  
(defstruct brain-activity 
  buffer
  value
  weight
  active
  s-time
  e-time)


(defvar *BRAIN-ACTIVITY* nil)
(defparameter *SAVE-BRAIN-ACTIVITY* t)


(defun get-buffer-activity (buffer record mstime save)
  (cond ((and save 
              (not (brain-activity-active buffer)) 
              (or (buffer-summary-busy record) (buffer-summary-request record)))
         (setf (brain-activity-active buffer) t)
         (setf (brain-activity-s-time buffer) (/ mstime 1000)))
        ((and (brain-activity-active buffer) 
              (or (not (buffer-summary-busy record)) (buffer-summary-busy->free record) (buffer-summary-error record)))
         (setf (brain-activity-active buffer) nil)
         (setf (brain-activity-e-time buffer) (/ mstime 1000))
         (setf (brain-activity-value buffer) (+ (brain-activity-value buffer)
                                          (* (brain-activity-weight buffer) 
                                             (- (brain-activity-e-time buffer) (brain-activity-s-time buffer))))))))


(defun get-time-series-brain-activity ()
  (let ((pb (make-brain-activity :buffer 'production :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (vb (make-brain-activity :buffer 'visual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (rb (make-brain-activity :buffer 'retrieval  :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (ib (make-brain-activity :buffer 'imaginal   :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
        (mb (make-brain-activity :buffer 'manual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
	(pre-time 0))

    (dolist (current-buffer-record (get-current-buffer-trace)) ;;buffer-record every mstime
      (let* ((mstime       (buffer-record-ms-time current-buffer-record))  
             (buffers      (buffer-record-buffers current-buffer-record)))

	(when (> (- mstime pre-time) 1000)
       
         (let ((value  (+ (brain-activity-value pb) 
                          (brain-activity-value vb) 
                          (brain-activity-value rb) 
                          (brain-activity-value ib) 
                          (brain-activity-value mb))))
            ;(format t "~% VALUE: ~f~%" value)
           (push (make-brain-activity-reference :total-value value
						:production (brain-activity-value pb)
						:visual (brain-activity-value vb)
						:retrieval (brain-activity-value rb)
						:imaginal (brain-activity-value ib)
						:manual (brain-activity-value mb)
						:time (/ mstime 1000)) *BRAIN-ACTIVITY*)
          
          ;;workload initiation
           (setf pb (make-brain-activity :buffer 'production :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
           (setf vb (make-brain-activity :buffer 'visual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
           (setf rb (make-brain-activity :buffer 'retrieval  :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
           (setf ib (make-brain-activity :buffer 'imaginal   :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0))
           (setf mb (make-brain-activity :buffer 'manual     :value 0 :weight 1 :active nil :s-time 0.0 :e-time 0.0)))

	 (setf pre-time mstime))


	;(format t "~% This part is performed?? time: ~a // ~a~%" mstime pre-time)
	
        (dolist (buffer buffers)
          (case (buffer-summary-name buffer)
              ((production) (get-buffer-activity pb buffer mstime *SAVE-BRAIN-ACTIVITY*))
              ((visual)     (get-buffer-activity vb buffer mstime *SAVE-BRAIN-ACTIVITY*))
              ((retrieval)  (get-buffer-activity rb buffer mstime *SAVE-BRAIN-ACTIVITY*))
              ((imaginal)   (get-buffer-activity ib buffer mstime *SAVE-BRAIN-ACTIVITY*))
              ((manual)     (get-buffer-activity mb buffer mstime *SAVE-BRAIN-ACTIVITY*))))))

     (let ((value  (+ (brain-activity-value pb) 
                      (brain-activity-value vb) 
                      (brain-activity-value rb) 
                      (brain-activity-value ib) 
                      (brain-activity-value mb))))
       (push (make-brain-activity-reference :total-value value
					    :production (brain-activity-value pb)
					    :visual (brain-activity-value vb)
					    :retrieval (brain-activity-value rb)
					    :imaginal (brain-activity-value ib)
					    :manual (brain-activity-value mb)
					    :time (mp-time)) *BRAIN-ACTIVITY*
					    ))))

;;;
;;; DECAY FUNCTION for brain acitivty
;;;
(defparameter *brain-decay* .156);0.156);.6)

(defun get-brain-activity-including-decay (reference decay ctime) ;;(WORKLOAD-DELAY 'MEMORY *WORKLOAD* (mp-time))
  (let ((total 0)
	(production 0)
	(visual 0)
	(retrieval 0)
	(imaginal 0)
	(manual 0))
    (dolist (ref reference)
      (let* ((rtime (- ctime (brain-activity-reference-time ref)))
	     (tbav (brain-activity-reference-total-value ref))
	     (pbav (brain-activity-reference-production ref))
	     (vbav (brain-activity-reference-visual ref))
	     (rbav (brain-activity-reference-retrieval ref))
	     (ibav (brain-activity-reference-imaginal ref))
	     (mbav (brain-activity-reference-manual ref))
	     (tv (if (= rtime 0) tbav (* tbav (expt rtime (* decay -1)))))
	     (pv (if (= rtime 0) pbav (* pbav (expt rtime (* decay -1)))))
	     (vv (if (= rtime 0) vbav (* vbav (expt rtime (* decay -1)))))
	     (rv (if (= rtime 0) rbav (* rbav (expt rtime (* decay -1)))))
	     (iv (if (= rtime 0) ibav (* ibav (expt rtime (* decay -1)))))
	     (mv (if (= rtime 0) mbav (* mbav (expt rtime (* decay -1))))))

	(when (> rtime 0)
	  (incf total tv)
	  (incf production pv)
	  (incf visual vv)
	  (incf retrieval rv)
	  (incf imaginal iv)
	  (incf manual mv))
	))
    (list total production visual retrieval imaginal manual)
    ))

(defun get-brain-activity-no-decay (reference ctime) 
  (let ((total 0)
	(production 0)
	(visual 0)
	(retrieval 0)
	(imaginal 0)
	(manual 0))
    (dolist (ref reference)
      (let* ((rtime (- ctime (brain-activity-reference-time ref)))
	     (tbav (brain-activity-reference-total-value ref))
	     (pbav (brain-activity-reference-production ref))
	     (vbav (brain-activity-reference-visual ref))
	     (rbav (brain-activity-reference-retrieval ref))
	     (ibav (brain-activity-reference-imaginal ref))
	     (mbav (brain-activity-reference-manual ref)))

	(when (> rtime 0)
	  (incf total tbav)
	  (incf production pbav)
	  (incf visual vbav)
	  (incf retrieval rbav)
	  (incf imaginal ibav)
	  (incf manual mbav))
	))
    (list total production visual retrieval imaginal manual)
    ))


(defun get-total-brain-activity ()
  ;;
  ;; 0. Total value
  ;; 1-1. Imaginal
  ;; 1-2. Visual
  ;; 1-3. Production
  ;; 1-4. Declarative
  ;; 2. Brain activity Time Series
  ;;
  
  (get-time-series-brain-activity)

  (let* ((activity (get-brain-activity-including-decay *BRAIN-ACTIVITY* *brain-decay* (mp-time)))
	 ;(activity (get-brain-activity-no-decay *BRAIN-ACTIVITY* (mp-time)))
	 (total (* 100 (/ (first activity) (* 5 (mp-time)))))
	 (imaginal (* 100 (/ (fifth activity) (mp-time))))
	 (visual (* 100 (/ (third activity) (mp-time))))
	 (production (* 100 (/ (second activity) (mp-time))))
	 (declarative (* 100 (/ (fourth activity) (mp-time))))
	 (time-series nil))

  (do ((i 0 (+ i 1)))
      ((> i (mp-time)) time-series)

    (let* ((activity-value (first (get-brain-activity-including-decay *BRAIN-ACTIVITY* *brain-decay* i)))
	   (no-decay-value (first (get-brain-activity-no-decay *BRAIN-ACTIVITY* i)))
	   (floor-value (if (realp activity-value) (floor activity-value) (floor no-decay-value))))

      (setf time-series (append time-series (list (list i floor-value)))))
	  

    ;(setf time-series (append time-series (list (list i (floor (first (get-brain-activity-including-decay *BRAIN-ACTIVITY* *brain-decay* i)))))))
    )
    (list total (list declarative imaginal production visual) time-series)
    ))


(add-act-r-command "get-total-brain-activity" 'get-total-brain-activity)




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Initialize the cognitive parameters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; 1. workload
;; 2. utility
;; 3. brain activity
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun init-cognitive-global-parameters ()

  (setf *WORKLOAD* nil)
  (setf *UTILITY* nil)
  (setf *BRAIN-ACTIVITY* nil)

  )

(add-act-r-command "init-cognitive-global-parameters" 'init-cognitive-global-parameters)
