#include "operators_p.h"
#include "operators.h"
#include "solver.h"
#include <stdlib.h>
#include <stdio.h>

static PetscReal default_rtol    = 1e-11;
static PetscInt  default_restart = 100;
static int       stab_added      = 0;

void steady_state(){
  PetscViewer    mat_view;
  PC             pc;
  Vec            x,b;
  KSP            ksp; /* linear solver context */
  int            row,col,its,j;
  PetscInt       i,Istart,Iend;
  PetscScalar    mat_tmp;
  long           dim;


  dim = total_levels*total_levels;

  if (!stab_added){
    if (nid==0) printf("Adding stabilization...\n");
    /*
     * Add elements to the matrix to make the normalization work
     * I have no idea why this works, I am copying it from qutip
     * We add 1.0 in the 0th spot and every n+1 after
     */
    if (nid==0) {
      row = 0;
      for (i=0;i<total_levels;i++){
        col = i*(total_levels+1);
        mat_tmp = 1.0 + 0.*PETSC_i;
        MatSetValue(full_A,row,col,mat_tmp,ADD_VALUES);
      }

      /* Print dense ham, if it was asked for */
      if (_print_dense_ham){
        FILE *fp_ham;

        fp_ham = fopen("ham","w");

        if (nid==0){
          for (i=0;i<total_levels;i++){
            for (j=0;j<total_levels;j++){
              fprintf(fp_ham,"%e ",_hamiltonian[i][j]);
            }
            fprintf(fp_ham,"\n");
          }
        }
        fclose(fp_ham);
      }
    }
  }

  
  MatGetOwnershipRange(full_A,&Istart,&Iend);
  /*
   * Explicitly add 0.0 to all diagonal elements;
   * this fixes a 'matrix in wrong state' message that PETSc
   * gives if the diagonal was never initialized.
   */
  if (nid==0) printf("Adding 0 to diagonal elements...\n");
  for (i=Istart;i<Iend;i++){
    mat_tmp = 0 + 0.*PETSC_i;
    MatSetValue(full_A,i,i,mat_tmp,ADD_VALUES);
  }

  
  /* Tell PETSc to assemble the matrix */
  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);
  if (nid==0) printf("Matrix Assembled.\n");
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view);
  PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO);
  MatView(full_A,mat_view);
  PetscViewerDestroy(&mat_view);
  /*
   * Create parallel vectors.
   * - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
   * we specify only the vector's global
   * dimension; the parallel partitioning is determined at runtime.
   * - Note: We form 1 vector from scratch and then duplicate as needed.
   */
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,dim);
  VecSetFromOptions(b);
  VecDuplicate(b,&x);

  /*
   * Set rhs, b, and solution, x to 1.0 in the first
   * element, 0.0 elsewhere.
   */
  VecSet(b,0.0);
  VecSet(x,0.0);
  
  if(nid==0) {
    row = 0;
    mat_tmp = 1.0 + 0.0*PETSC_i;
    VecSetValue(x,row,mat_tmp,INSERT_VALUES);
    VecSetValue(b,row,mat_tmp,INSERT_VALUES);
  }
  
  /* Assemble x and b */
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*
     *           Create the linear solver and set various options         *
     *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
   * Create linear solver context
   */
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  /*
   * Set operators. Here the matrix that defines the linear system
   * also serves as the preconditioning matrix.
   */
  KSPSetOperators(ksp,full_A,full_A);
  
  /*
   * Set good default options for solver
   */
  /* relative tolerance */
  KSPSetTolerances(ksp,default_rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

  /* bjacobi preconditioner */
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCASM);

  /* gmres solver with 100 restart*/
  KSPSetType(ksp,KSPGMRES);
  KSPGMRESSetRestart(ksp,default_restart);
  /* 
   * Set runtime options, e.g.,
   *     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
   */
  KSPSetFromOptions(ksp);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nid==0) printf("KSP set. Solving for steady state...\n");
  KSPSolve(ksp,b,x);
  get_populations(x);

  KSPGetIterationNumber(ksp,&its);

  PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its);

  /* Free work space */
  KSPDestroy(&ksp);
  VecDestroy(&x);
  VecDestroy(&b);

  return;
}

void get_populations(Vec x) {
  int         j,my_levels,n_after,cur_state,num_pop;
  int         *i_sub_to_i_pop;
  PetscInt    x_low,x_high,i;
  PetscScalar *xa;
  PetscReal   tmp_real;
  double      *populations;
  VecGetOwnershipRange(x,&x_low,&x_high);
  VecGetArray(x,&xa); 

  /*
   * Loop through operators to see how many populations we need to 
   * calculate, because VEC need a population for each level.
   * We also set an array that translates i_subsystem to i_pop for normal ops.
   */
  i_sub_to_i_pop = malloc(num_subsystems*sizeof(int));
  num_pop = 0;
  for (i=0;i<num_subsystems;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      i_sub_to_i_pop[i] = num_pop;
      num_pop += subsystem_list[i]->my_levels;
    } else {
      i_sub_to_i_pop[i] = num_pop;
      num_pop += 1;      
    }
  }

  /* Initialize population arrays */
  populations = malloc(num_pop*sizeof(double));
  for (i=0;i<num_pop;i++){
    populations[i] = 0.0;
  }


  for (i=0;i<total_levels;i++){
    if ((i*total_levels+i)>=x_low&&(i*total_levels+i)<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[i*(total_levels)+i-x_low]);
      //      printf("%e \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need 
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = mod(mod(floor(i/n_a),l*n_a),l)
         * For regular operators, these are just number operators, and we count from 0,
         * so, cur_state = i_subspace
         *
         * For VEC ops, we can use the same technique. Once we get cur_state (i_subspace),
         * we use that to go the appropriate location in the population array.
         */
        if (subsystem_list[j]->my_op_type==VEC){
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]+cur_state] += tmp_real;
        } else {
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]] += tmp_real*cur_state;
        }
      }
    }
  } 



  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(populations,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  if(nid==0) {
    printf("Populations: ");
    for(i=0;i<num_pop;i++){
      printf(" %e ",populations[i]);
    }
    printf("\n");
  }

  /* Put the array back in Petsc's hands */
  VecRestoreArray(x,&xa);
  /* Free memory */
  free(i_sub_to_i_pop);
  free(populations);
  return;
}
