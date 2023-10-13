
SELECT name, salary, managerId
FROM Employee t1
LEFT JOIN(
    SELECT id, salary
    FROM Employee
) t2
ON t1.managerId = t2.id
;
