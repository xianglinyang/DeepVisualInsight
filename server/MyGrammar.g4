grammar MyGrammar;           
// parser rules
expr : ACTION ('where' cond1)*;         
cond1 : cond2 (multiplecond2)* 
      | '(' parencond1 ')' (multiplecond2)*
      ;         
parencond1: cond1;
multiplecond2: CONDOP (cond2|cond1);
cond2 : parameter OP parameter;         
parameter : STRING | number | array;
array: '(' INT (',' INT)* ')';
number: INT         # Positive
      | '-' INT     # Negative
      ;

// lexer rules
ACTION : 'search for samples';
STRING : [A-Za-z.]+ ;     
OP: '=' | '!=' | '==';
CONDOP: '&' | '|'; 
INT : [0-9]+;
WS : [ '\t\r\n]+ -> skip ; 
