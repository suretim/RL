

set(component_srcs 
	"ai.c"
	"ai_log.c"
	"ml_grade.c"
	"ml_dynamic_sun.c"
	"ml_pid.c"
	"ml_rule.c"
	"ai_insight.c"
	"ni_debug.c"
	"pso.c" 
)


idf_component_register(
	SRCS "${component_srcs}"

	INCLUDE_DIRS .

	REQUIRES "driver"
	PRIV_REQUIRES "bsp"
	PRIV_REQUIRES "func"
	PRIV_REQUIRES "plant_env"	
	PRIV_REQUIRES "devices"
	PRIV_REQUIRES "wifi"	
)