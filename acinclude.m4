dnl -*- tab-width: 2 -*-
AC_DEFUN(AC_PROG_PYTHON,
	[ AC_CHECK_PROG(PYTHON, python, python) ])

AC_DEFUN(AC_PYTHON_INCLUDE_PATH,
	[ AC_MSG_CHECKING(python include directory)
		changequote(, )dnl
		ac_l_bracket="[" ac_r_bracket="]"  # trying to outsmart Autoconf here.
		changequote([, ])dnl
		echo "import sys; print sys.prefix + \"/include/python\" + sys.version$ac_l_bracket:3$ac_r_bracket" > conftest
		ac_cv_py_inc_path=`${PYTHON-python} conftest`
		PYTHON_INCLUDE_PATH=$ac_cv_py_inc_path
		AC_MSG_RESULT($ac_cv_py_inc_path)
		AC_SUBST(PYTHON_INCLUDE_PATH)
	])

