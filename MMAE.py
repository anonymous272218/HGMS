# The weight for ROUGE-L, MAXsim, and IP is 1.641,0.854, 0.806 respectively and the intercept is 1.978.

ROUGE = 0.4083
IP = 0.7895
Max = 0.4581

a = ROUGE * 1.641 + Max*0.854 + IP*0.806 + 1.978
b = 0.5319 * 1.641 + 0.854 + 0.806 + 1.978
print(a/b)

# MR = 0.0
# a = ROUGE * 1.54 + Max*0.42 + IP*1.25 + MR * 0.98 + 1.4
# b = 0.5319 * 1.54 + 1*0.42 + 1 * 1.25 + 1* 0.98 + 1.4
# print(a/b)
