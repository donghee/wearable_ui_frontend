
;;;;;;;;;;;;
;; 
;; GAIT SIMPLE MODEL - New version
;;
;; used buffers :: goal, visual, manual, retrieval buffer
;;
;;;;;;;;;;;;

(clear-all)

(define-model gait-model
    
  (sgp :v t :esc t :egs 0.0 :ul t :ult t)
  ;;(sgp :lf .1)
  (sgp :show-focus t :trace-detail low)
  
  (sgp :save-buffer-trace t)

  (sgp :ul t :ult t)
  ;(sgp :utility-offsets utility-output)
  ;(sgp :utility-hook utility-switch)
  
  (sgp :bold-inc .1)
  
  ;;; SGP of buffer/module history
  (sgp :save-p-history t)
  (sgp :save-dm-history t)
  (sgp :save-buffer-history t)
  
  ;(sgp :set-resistance-threshold 0.9)
  
  
  ;;goal chunk type
  (chunk-type gait-move walk state task)
  
  ;;imaginal chunk type
  (chunk-type check-task walk force)

  ;;visual position type
  (chunk-type visual-pos x-pos y-pos value)
  
  (add-dm
   (first-goal isa gait-move walk 0 state nil task gait)
   (visual-pos-o isa visual-pos x-pos 50 y-pos 50 value "o")
   (visual-pos-x isa visual-pos x-pos 50 y-pos 75 value "x")
   )
  
  ;;first-movement
  (P do-start
     =goal>
       state nil
   ==>
     =goal>
       state search
     !eval! (start-device-for-gait)
       )

  (P search-feedforward
     =goal>
       state search
   ==>
     =goal>
       state attend
     +visual-location>
       value "o"
       )

  (P attend-feedforward
     =goal>
       state attend
     =visual-location>
   ==>
     =goal>
       state make-swing
     +visual>
       cmd move-attention
       screen-pos =visual-location
     )

  (P do-swing
     =goal>
       state make-swing
     =visual>
   ==>
     =goal>
       state check
     -visual>
     +visual-location>
       value "o"
     +manual>
       cmd gait-movement
       stride 5
       )

  (P attend-feedback
     =goal>
       state check
     =visual-location>
     ?manual>
       state busy
   ==>
     =goal>
       state attend-check
     +visual>
       cmd move-attention
       screen-pos =visual-location
     +retrieval>
       isa visual-pos  
       x-pos 50
       y-pos 50
       )

  (P movement-continuation
     =goal>
       state attend-check
     =visual>
       isa      visual-object
       value    =val  
     =retrieval>
       isa visual-pos
       value   =val
   ==>
     =goal>
       state check
       )

  (P movement-re-control
     =goal>
       state attend-check
     =visual>
       isa      visual-object
       value    =val  
     =retrieval>
       isa visual-pos
     - value   =val
   ==>
     =goal>
       state check
     +visual-location>
       value "o"
       )

  (P find-device-error
     =goal>
       state check
     ?manual>
       state busy
     ?wearable-device>
       state error  
   ==>
     =goal>
       state check
     +visual-location>
       value "x"
     ;+manual>
     ;cmd respond-to-gait-error
     !eval! (set-error-for-gait-device)
       )

  (P do-support-state
     =goal>
       state check
     ?manual>
       state free
   ==>
     =goal>
       state search
       )

     
     
     
     
  

 
  (goal-focus first-goal)
  
  ;(spp attend-an-letter :u 5)
  ;(spp check-the-task   :u 10)
  ;(spp check-intuitive-color :u 10)
  ;(spp check-intuitive-letter :u 10)
  
  ;(spp action-incongruent-stroop :reward 20)
  ;(spp action-congruent-stroop :reward 0)
    
    )
