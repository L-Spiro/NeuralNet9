
/* A Bison CXmlParser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison LALR(1) parsers in C++
   
      Copyright (C) 2002, 2003, 2004, 2005, 2006, 2007, 2008 Free Software
   Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison CXmlParser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   CXmlParser generator using the skeleton or a modified version thereof
   as a CXmlParser skeleton.  Alternatively, if you modify or redistribute
   the CXmlParser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* First part of user declarations.  */


/*
 * WARNING: This is not an XML 1.0 CXmlParser, but an experiment with an XML-like
 * language. See http://www.w3.org/XML/9707/XML-in-C
 *
 * Author: Bert Bos <bert@w3.org>
 * Created: 9 July 1997
 *
 * Copyright C 1997-2004 World Wide Web Consortium
 * See http://www.w3.org/Consortium/Legal/copyright-software-19980720.html
 */

#include "../LSXml.h"
#include "../LSXmlContainer.h"
#include "../LSXmlLexer.h"
#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// MACROS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#define YYINCLUDED_STDLIB_H

//extern char yytext[];
// Announce to Flex the prototype we want for lexing function.
extern int yylex( /*YYSTYPE*/void * _pvNodeUnion, lsx::CXmlLexer * _pxlLexer );

#include "../LSXmlSyntaxNodes.h"





#include "LSXmlParser.h"

/* User implementation prologue.  */



#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* FIXME: INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#define YYUSE(e) ((void) (e))

/* Enable debugging if requested.  */
#if YYDEBUG

/* A pseudo ostream that takes yydebug_ into account.  */
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)	\
do {							\
  if (yydebug_)						\
    {							\
      *yycdebug_ << Title << ' ';			\
      yy_symbol_print_ ((Type), (Value), (Location));	\
      *yycdebug_ << std::endl;				\
    }							\
} while (false)

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug_)				\
    yy_reduce_print_ (Rule);		\
} while (false)

# define YY_STACK_PRINT()		\
do {					\
  if (yydebug_)				\
    yystack_print_ ();			\
} while (false)

#else /* !YYDEBUG */

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_REDUCE_PRINT(Rule)
# define YY_STACK_PRINT()

#endif /* !YYDEBUG */

#define yyerrok		(yyerrstatus_ = 0)
#define yyclearin	(yychar = yyempty_)

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)



namespace yy {

#if YYERROR_VERBOSE

  /* Return YYSTR after stripping away unnecessary quotes and
     backslashes, so that it's suitable for yyerror.  The heuristic is
     that double-quoting is unnecessary unless the string contains an
     apostrophe, a comma, or backslash (other than backslash-backslash).
     YYSTR is taken from yytname.  */
  std::string
  CXmlParser::yytnamerr_ (const char *yystr)
  {
    if (*yystr == '"')
      {
        std::string yyr = "";
        char const *yyp = yystr;

        for (;;)
          switch (*++yyp)
            {
            case '\'':
            case ',':
              goto do_not_strip_quotes;

            case '\\':
              if (*++yyp != '\\')
                goto do_not_strip_quotes;
              /* Fall through.  */
            default:
              yyr += *yyp;
              break;

            case '"':
              return yyr;
            }
      do_not_strip_quotes: ;
      }

    return yystr;
  }

#endif

  /// Build a CXmlParser object.
  CXmlParser::CXmlParser (class CXmlLexer * m_peelLexer_yyarg, class CXmlContainer * m_peecContainer_yyarg)
    :
#if YYDEBUG
      yydebug_ (false),
      yycdebug_ (&std::cerr),
#endif
      m_peelLexer (m_peelLexer_yyarg),
      m_peecContainer (m_peecContainer_yyarg)
  {
  }

  CXmlParser::~CXmlParser ()
  {
  }

#if YYDEBUG
  /*--------------------------------.
  | Print this symbol on YYOUTPUT.  |
  `--------------------------------*/

  inline void
  CXmlParser::yy_symbol_value_print_ (int yytype,
			   const semantic_type* yyvaluep, const location_type* yylocationp)
  {
    YYUSE (yylocationp);
    YYUSE (yyvaluep);
    switch (yytype)
      {
         default:
	  break;
      }
  }


  void
  CXmlParser::yy_symbol_print_ (int yytype,
			   const semantic_type* yyvaluep, const location_type* yylocationp)
  {
    *yycdebug_ << (yytype < yyntokens_ ? "token" : "nterm")
	       << ' ' << yytname_[yytype] << " ("
	       << *yylocationp << ": ";
    yy_symbol_value_print_ (yytype, yyvaluep, yylocationp);
    *yycdebug_ << ')';
  }
#endif

  void
  CXmlParser::yydestruct_ (const char* yymsg,
			   int yytype, semantic_type* yyvaluep, location_type* yylocationp)
  {
    YYUSE (yylocationp);
    YYUSE (yymsg);
    YYUSE (yyvaluep);

    YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

    switch (yytype)
      {
  
	default:
	  break;
      }
  }

  void
  CXmlParser::yypop_ (unsigned int n)
  {
    yystate_stack_.pop (n);
    yysemantic_stack_.pop (n);
    yylocation_stack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
  CXmlParser::debug_stream () const
  {
    return *yycdebug_;
  }

  void
  CXmlParser::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


  CXmlParser::debug_level_type
  CXmlParser::debug_level () const
  {
    return yydebug_;
  }

  void
  CXmlParser::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif

  int
  CXmlParser::parse ()
  {
    /// Lookahead and lookahead in internal form.
    int yychar = yyempty_;
    int yytoken = 0;

    /* State.  */
    int yyn;
    int yylen = 0;
    int yystate = 0;

    /* Error handling.  */
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// Semantic value of the lookahead.
    semantic_type yylval;
    /// Location of the lookahead.
    location_type yylloc;
    /// The locations where the error started and ended.
    location_type yyerror_range[2];

    /// $$.
    semantic_type yyval;
    /// @$.
    location_type yyloc;

    int yyresult;

    YYCDEBUG << "Starting parse" << std::endl;


    /* Initialize the stacks.  The initial state will be pushed in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystate_stack_ = state_stack_type (0);
    yysemantic_stack_ = semantic_stack_type (0);
    yylocation_stack_ = location_stack_type (0);
    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yylloc);

    /* New state.  */
  yynewstate:
    yystate_stack_.push (yystate);
    YYCDEBUG << "Entering state " << yystate << std::endl;

    /* Accept?  */
    if (yystate == yyfinal_)
      goto yyacceptlab;

    goto yybackup;

    /* Backup.  */
  yybackup:

    /* Try to take a decision without lookahead.  */
    yyn = yypact_[yystate];
    if (yyn == yypact_ninf_)
      goto yydefault;

    /* Read a lookahead token.  */
    if (yychar == yyempty_)
      {
	YYCDEBUG << "Reading a token: ";
	yychar = yylex (&yylval, m_peelLexer);
      }


    /* Convert token to internal form.  */
    if (yychar <= yyeof_)
      {
	yychar = yytoken = yyeof_;
	YYCDEBUG << "Now at end of input." << std::endl;
      }
    else
      {
	yytoken = yytranslate_ (yychar);
	YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
      }

    /* If the proper action on seeing token YYTOKEN is to reduce or to
       detect an error, take that action.  */
    yyn += yytoken;
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yytoken)
      goto yydefault;

    /* Reduce or error.  */
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
	if (yyn == 0 || yyn == yytable_ninf_)
	goto yyerrlab;
	yyn = -yyn;
	goto yyreduce;
      }

    /* Shift the lookahead token.  */
    YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

    /* Discard the token being shifted.  */
    yychar = yyempty_;

    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yylloc);

    /* Count tokens shifted since error; after three, turn off error
       status.  */
    if (yyerrstatus_)
      --yyerrstatus_;

    yystate = yyn;
    goto yynewstate;

  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[yystate];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;

  /*-----------------------------.
  | yyreduce -- Do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    /* If YYLEN is nonzero, implement the default value of the action:
       `$$ = $1'.  Otherwise, use the top of the stack.

       Otherwise, the following line sets YYVAL to garbage.
       This behavior is undocumented and Bison
       users should not rely upon it.  */
    if (yylen)
      yyval = yysemantic_stack_[yylen - 1];
    else
      yyval = yysemantic_stack_[0];

    {
      slice<location_type, location_stack_type> slice (yylocation_stack_, yylen);
      YYLLOC_DEFAULT (yyloc, slice, yylen);
    }
    YY_REDUCE_PRINT (yyn);
    switch (yyn)
      {
	  case 2:

    { (yyval.sStringIndex) = m_peecContainer->AddString( m_peelLexer->YYText() ); }
    break;

  case 3:

    { (yyval.sStringIndex) = m_peecContainer->AddValue( m_peelLexer->YYText() ); }
    break;

  case 4:

    { (yyval.sStringIndex) = m_peecContainer->AddString( m_peelLexer->YYText() ); }
    break;

  case 5:

    { (yyval.sStringIndex) = m_peecContainer->AddAttributeStart( m_peelLexer->YYText() ); }
    break;

  case 6:

    { m_peecContainer->AddDocument( (yyval.nNode), (yysemantic_stack_[(3) - (1)].nNode), (yysemantic_stack_[(3) - (2)].nNode), (yysemantic_stack_[(3) - (3)].nNode) ); }
    break;

  case 7:

    { m_peecContainer->AddProlog( (yyval.nNode), (yysemantic_stack_[(3) - (1)].nNode), (yysemantic_stack_[(3) - (2)].nNode), (yysemantic_stack_[(3) - (3)].nNode) ); }
    break;

  case 8:

    { m_peecContainer->AddVersion( (yyval.nNode), m_peelLexer->YYText() ); }
    break;

  case 9:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 10:

    { m_peecContainer->AddEncoding( (yyval.nNode), m_peelLexer->YYText() ); }
    break;

  case 11:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 12:

    { m_peecContainer->AddMiscSeq( (yyval.nNode), (yysemantic_stack_[(2) - (1)].nNode), (yysemantic_stack_[(2) - (2)].nNode) ); }
    break;

  case 13:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 14:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 15:

    { (yyval.nNode) = (yysemantic_stack_[(1) - (1)].nNode); }
    break;

  case 16:

    { m_peecContainer->AddAttributeDecl( (yyval.nNode), (yysemantic_stack_[(4) - (2)].sStringIndex), (yysemantic_stack_[(4) - (3)].nNode) ); }
    break;

  case 17:

    { m_peecContainer->AddElement( (yyval.nNode), (yysemantic_stack_[(3) - (1)].sStringIndex), (yysemantic_stack_[(3) - (2)].nNode), (yysemantic_stack_[(3) - (3)].nNode) ); }
    break;

  case 18:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 19:

    { m_peecContainer->AddContent( (yyval.nNode), (yysemantic_stack_[(5) - (2)].nNode), (yysemantic_stack_[(5) - (4)].sStringIndex) ); }
    break;

  case 20:

    { m_peecContainer->AddContentData( (yyval.nNode), (yysemantic_stack_[(2) - (1)].nNode), (yysemantic_stack_[(2) - (2)].sStringIndex) ); }
    break;

  case 21:

    { m_peecContainer->AddContentMisc( (yyval.nNode), (yysemantic_stack_[(2) - (1)].nNode), (yysemantic_stack_[(2) - (2)].nNode) ); }
    break;

  case 22:

    { m_peecContainer->AddContentElement( (yyval.nNode), (yysemantic_stack_[(2) - (1)].nNode), (yysemantic_stack_[(2) - (2)].nNode) ); }
    break;

  case 23:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 24:

    { (yyval.sStringIndex) = (yysemantic_stack_[(1) - (1)].sStringIndex); }
    break;

  case 25:

    { (yyval.sStringIndex) = size_t( -1 ); }
    break;

  case 26:

    { m_peecContainer->AddAttributeList( (yyval.nNode), (yysemantic_stack_[(2) - (1)].nNode), (yysemantic_stack_[(2) - (2)].nNode) ); }
    break;

  case 27:

    { m_peecContainer->AddEmpty( (yyval.nNode) ); }
    break;

  case 28:

    { m_peecContainer->AddAttribute( (yyval.nNode), (yysemantic_stack_[(1) - (1)].sStringIndex) ); }
    break;

  case 29:

    { m_peecContainer->AddAttribute( (yyval.nNode), (yysemantic_stack_[(3) - (1)].sStringIndex), (yysemantic_stack_[(3) - (3)].sStringIndex) ); }
    break;



	default:
          break;
      }
    YY_SYMBOL_PRINT ("-> $$ =", yyr1_[yyn], &yyval, &yyloc);

    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();

    yysemantic_stack_.push (yyval);
    yylocation_stack_.push (yyloc);

    /* Shift the result of the reduction.  */
    yyn = yyr1_[yyn];
    yystate = yypgoto_[yyn - yyntokens_] + yystate_stack_[0];
    if (0 <= yystate && yystate <= yylast_
	&& yycheck_[yystate] == yystate_stack_[0])
      yystate = yytable_[yystate];
    else
      yystate = yydefgoto_[yyn - yyntokens_];
    goto yynewstate;

  /*------------------------------------.
  | yyerrlab -- here on detecting error |
  `------------------------------------*/
  yyerrlab:
    /* If not already recovering from an error, report this error.  */
    if (!yyerrstatus_)
      {
	++yynerrs_;
	error (yylloc, yysyntax_error_ (yystate));
      }

    yyerror_range[0] = yylloc;
    if (yyerrstatus_ == 3)
      {
	/* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

	if (yychar <= yyeof_)
	  {
	  /* Return failure if at end of input.  */
	  if (yychar == yyeof_)
	    YYABORT;
	  }
	else
	  {
	    yydestruct_ ("Error: discarding", yytoken, &yylval, &yylloc);
	    yychar = yyempty_;
	  }
      }

    /* Else will try to reuse lookahead token after shifting the error
       token.  */
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:

    /* Pacify compilers like GCC when the user code never invokes
       YYERROR and the label yyerrorlab therefore never appears in user
       code.  */
    if (false)
      goto yyerrorlab;

    yyerror_range[0] = yylocation_stack_[yylen - 1];
    /* Do not reclaim the symbols of the rule which action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    yystate = yystate_stack_[0];
    goto yyerrlab1;

  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;	/* Each real token shifted decrements this.  */

    for (;;)
      {
	yyn = yypact_[yystate];
	if (yyn != yypact_ninf_)
	{
	  yyn += yyterror_;
	  if (0 <= yyn && yyn <= yylast_ && yycheck_[yyn] == yyterror_)
	    {
	      yyn = yytable_[yyn];
	      if (0 < yyn)
		break;
	    }
	}

	/* Pop the current state because it cannot handle the error token.  */
	if (yystate_stack_.height () == 1)
	YYABORT;

	yyerror_range[0] = yylocation_stack_[0];
	yydestruct_ ("Error: popping",
		     yystos_[yystate],
		     &yysemantic_stack_[0], &yylocation_stack_[0]);
	yypop_ ();
	yystate = yystate_stack_[0];
	YY_STACK_PRINT ();
      }

    yyerror_range[1] = yylloc;
    // Using YYLLOC is tempting, but would change the location of
    // the lookahead.  YYLOC is available though.
    YYLLOC_DEFAULT (yyloc, (yyerror_range - 1), 2);
    yysemantic_stack_.push (yylval);
    yylocation_stack_.push (yyloc);

    /* Shift the error token.  */
    YY_SYMBOL_PRINT ("Shifting", yystos_[yyn],
		     &yysemantic_stack_[0], &yylocation_stack_[0]);

    yystate = yyn;
    goto yynewstate;

    /* Accept.  */
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;

    /* Abort.  */
  yyabortlab:
    yyresult = 1;
    goto yyreturn;

  yyreturn:
    if (yychar != yyempty_)
      yydestruct_ ("Cleanup: discarding lookahead", yytoken, &yylval, &yylloc);

    /* Do not reclaim the symbols of the rule which action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    while (yystate_stack_.height () != 1)
      {
	yydestruct_ ("Cleanup: popping",
		   yystos_[yystate_stack_[0]],
		   &yysemantic_stack_[0],
		   &yylocation_stack_[0]);
	yypop_ ();
      }

    return yyresult;
  }

  // Generate an error message.
  std::string
  CXmlParser::yysyntax_error_ (int yystate)
  {
    std::string res;
    YYUSE (yystate);
#if YYERROR_VERBOSE
    int yyn = yypact_[yystate];
    if (yypact_ninf_ < yyn && yyn <= yylast_)
      {
	/* Start YYX at -YYN if negative to avoid negative indexes in
	   YYCHECK.  */
	int yyxbegin = yyn < 0 ? -yyn : 0;

	/* Stay within bounds of both yycheck and yytname.  */
	int yychecklim = yylast_ - yyn + 1;
	int yyxend = yychecklim < yyntokens_ ? yychecklim : yyntokens_;
	int count = 0;
	for (int x = yyxbegin; x < yyxend; ++x)
	  if (yycheck_[x + yyn] == x && x != yyterror_)
	    ++count;

	// FIXME: This method of building the message is not compatible
	// with internationalization.  It should work like yacc.c does it.
	// That is, first build a string that looks like this:
	// error, "Syntax error, error, unexpected %s or %s or %s"
	// Then, invoke YY_ on this string.
	// Finally, use the string as a format to output
	// yytname_[tok], etc.
	// Until this gets fixed, this message appears in English only.
	res = error, "Syntax error, error, unexpected ";
	res += yytnamerr_ (yytname_[tok]);
	if (count < 5)
	  {
	    count = 0;
	    for (int x = yyxbegin; x < yyxend; ++x)
	      if (yycheck_[x + yyn] == x && x != yyterror_)
		{
		  res += (!count++) ? ", expecting " : " or ";
		  res += yytnamerr_ (yytname_[x]);
		}
	  }
      }
    else
#endif
      res = YY_("Syntax error.");
    return res;
  }


  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
  const signed char CXmlParser::yypact_ninf_ = -15;
  const signed char
  CXmlParser::yypact_[] =
  {
         0,   -15,     1,   -11,     7,   -15,   -15,   -15,   -15,   -15,
     -15,     8,    -2,    -2,    10,   -15,   -15,    14,   -15,   -15,
      -3,   -15,   -15,   -15,   -15,    -4,     9,   -15,    -3,   -15,
     -15,   -15,   -15,   -15,   -15,     2,   -15,    15,   -15,   -15
  };

  /* YYDEFACT[S] -- default rule to reduce with in state S when YYTABLE
     doesn't specify something else to do.  Zero means the default is an
     error.  */
  const unsigned char
  CXmlParser::yydefact_[] =
  {
         9,     8,     0,     0,    11,     1,     5,    27,    13,    10,
      13,     0,     6,     7,     0,    23,     2,    28,    17,    26,
       0,    14,    12,    15,    18,     0,     0,    27,    25,     4,
      20,    21,    22,     3,    29,     0,    24,     0,    16,    19
  };

  /* YYPGOTO[NTERM-NUM].  */
  const signed char
  CXmlParser::yypgoto_[] =
  {
       -15,   -14,   -15,   -15,   -15,   -15,   -15,   -15,   -15,    12,
      -1,   -15,     3,   -15,   -15,   -15,     4,   -15
  };

  /* YYDEFGOTO[NTERM-NUM].  */
  const signed char
  CXmlParser::yydefgoto_[] =
  {
        -1,    17,    34,    30,     7,     2,     3,     4,    10,    12,
      22,    23,     8,    18,    25,    37,    11,    19
  };

  /* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule which
     number is the opposite.  If zero, do what YYDEFACT says.  */
  const signed char CXmlParser::yytable_ninf_ = -1;
  const unsigned char
  CXmlParser::yytable_[] =
  {
        20,     5,    20,     1,     6,    28,    27,    38,    16,    29,
      21,     6,    21,    16,    36,    14,    15,     9,    24,    16,
      26,    33,    13,    39,    31,     0,     0,     0,    32,     0,
       0,    35
  };

  /* YYCHECK.  */
  const signed char
  CXmlParser::yycheck_[] =
  {
         4,     0,     4,     3,    15,     9,    20,     5,    11,    13,
      14,    15,    14,    11,    28,     7,     8,    10,     8,    11,
       6,    12,    10,     8,    25,    -1,    -1,    -1,    25,    -1,
      -1,    27
  };

  /* STOS_[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
  const unsigned char
  CXmlParser::yystos_[] =
  {
         0,     3,    21,    22,    23,     0,    15,    20,    28,    10,
      24,    32,    25,    25,     7,     8,    11,    17,    29,    33,
       4,    14,    26,    27,     8,    30,     6,    17,     9,    13,
      19,    26,    28,    12,    18,    32,    17,    31,     5,     8
  };

#if YYDEBUG
  /* TOKEN_NUMBER_[YYLEX-NUM] -- Internal symbol number corresponding
     to YYLEX-NUM.  */
  const unsigned short int
  CXmlParser::yytoken_number_[] =
  {
         0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270
  };
#endif

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
  const unsigned char
  CXmlParser::yyr1_[] =
  {
         0,    16,    17,    18,    19,    20,    21,    22,    23,    23,
      24,    24,    25,    25,    26,    26,    27,    28,    29,    29,
      30,    30,    30,    30,    31,    31,    32,    32,    33,    33
  };

  /* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
  const unsigned char
  CXmlParser::yyr2_[] =
  {
         0,     2,     1,     1,     1,     1,     3,     3,     1,     0,
       1,     0,     2,     0,     1,     1,     4,     3,     2,     5,
       2,     2,     2,     0,     1,     0,     2,     0,     1,     3
  };

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
  /* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
     First, the terminals, then, starting at \a yyntokens_, nonterminals.  */
  const char*
  const CXmlParser::yytname_[] =
  {
    "$end", "error", "$undefined", "LSX_VERSION", "LSX_ATTDEF",
  "LSX_ENDDEF", "LSX_EQ", "LSX_SLASH", "LSX_CLOSE", "LSX_END",
  "LSX_ENCODING", "LSX_NAME", "LSX_VALUE", "LSX_DATA", "LSX_COMMENT",
  "LSX_START", "$accept", "name", "value", "data", "start", "document",
  "prolog", "version_opt", "encoding_opt", "misc_seq_opt", "misc",
  "attribute_decl", "element", "empty_or_content", "content", "name_opt",
  "attribute_seq_opt", "attribute", 0
  };
#endif

#if YYDEBUG
  /* YYRHS -- A `-1'-separated list of the rules' RHS.  */
  const CXmlParser::rhs_number_type
  CXmlParser::yyrhs_[] =
  {
        21,     0,    -1,    11,    -1,    12,    -1,    13,    -1,    15,
      -1,    22,    28,    25,    -1,    23,    24,    25,    -1,     3,
      -1,    -1,    10,    -1,    -1,    25,    26,    -1,    -1,    14,
      -1,    27,    -1,     4,    17,    32,     5,    -1,    20,    32,
      29,    -1,     7,     8,    -1,     8,    30,     9,    31,     8,
      -1,    30,    19,    -1,    30,    26,    -1,    30,    28,    -1,
      -1,    17,    -1,    -1,    32,    33,    -1,    -1,    17,    -1,
      17,     6,    18,    -1
  };

  /* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
     YYRHS.  */
  const unsigned char
  CXmlParser::yyprhs_[] =
  {
         0,     0,     3,     5,     7,     9,    11,    15,    19,    21,
      22,    24,    25,    28,    29,    31,    33,    38,    42,    45,
      51,    54,    57,    60,    61,    63,    64,    67,    68,    70
  };

  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
  const unsigned char
  CXmlParser::yyrline_[] =
  {
         0,    64,    64,    68,    72,    76,    80,    83,    87,    88,
      91,    92,    95,    96,    99,   100,   103,   107,   112,   113,
     117,   118,   119,   120,   123,   124,   127,   128,   131,   132
  };

  // Print the state stack on the debug stream.
  void
  CXmlParser::yystack_print_ ()
  {
    *yycdebug_ << "Stack now";
    for (state_stack_type::const_iterator i = yystate_stack_.begin ();
	 i != yystate_stack_.end (); ++i)
      *yycdebug_ << ' ' << *i;
    *yycdebug_ << std::endl;
  }

  // Report on the debug stream that the rule \a yyrule is going to be reduced.
  void
  CXmlParser::yy_reduce_print_ (int yyrule)
  {
    unsigned int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    /* Print the symbols being reduced, and their result.  */
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
	       << " (line " << yylno << "):" << std::endl;
    /* The symbols being reduced.  */
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
		       yyrhs_[yyprhs_[yyrule] + yyi],
		       &(yysemantic_stack_[(yynrhs) - (yyi + 1)]),
		       &(yylocation_stack_[(yynrhs) - (yyi + 1)]));
  }
#endif // YYDEBUG

  /* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
  CXmlParser::token_number_type
  CXmlParser::yytranslate_ (int t)
  {
    static
    const token_number_type
    translate_table[] =
    {
           0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15
    };
    if ((unsigned int) t <= yyuser_token_number_max_)
      return translate_table[t];
    else
      return yyundef_token_;
  }

  const int CXmlParser::yyeof_ = 0;
  const int CXmlParser::yylast_ = 31;
  const int CXmlParser::yynnts_ = 18;
  const int CXmlParser::yyempty_ = -2;
  const int CXmlParser::yyfinal_ = 5;
  const int CXmlParser::yyterror_ = 1;
  const int CXmlParser::yyerrcode_ = 256;
  const int CXmlParser::yyntokens_ = 16;

  const unsigned int CXmlParser::yyuser_token_number_max_ = 270;
  const CXmlParser::token_number_type CXmlParser::yyundef_token_ = 2;



} // yy





int yylex( /*YYSTYPE*/void * /*_pvNodeUnion*/, lsx::CXmlLexer * _pxlLexer ) {
	return _pxlLexer->yylex();
}

void yy::CXmlParser::error( const yy::location &/*_lLoc*/, const std::string &/*_strM*/ ) {

}
